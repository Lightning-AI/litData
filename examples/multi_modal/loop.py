"""Train, validate and test loop using the lightning framework."""

import logging
import os
from collections.abc import Sequence
from typing import Any, Union

import lightning as pl
import pandas as pd
import torch
from dataloader import EncoderAndTokenizer
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from model_arc import BertResNetClassifier
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from transformers import AdamW

logger = logging.getLogger()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
seed = seed_everything(21, workers=True)


class SklearnMetricsCallback(pl.Callback):
    def __init__(self, label_encoder, hyperparameters):
        super().__init__()
        self.label_encoder = label_encoder
        self.hyperparameters = hyperparameters

    def on_test_epoch_end(self, trainer, pl_module):
        output_list = []
        ground_truths_list = []
        numerical_ids_list = []
        for batch_item in pl_module.pred_list:
            output_list.append(batch_item["outputs"])
            ground_truths_list.append(batch_item["ground_truth"])
            numerical_ids_list.append(batch_item["numerical_id"])
        output_tensor = torch.concat(output_list, dim=0)
        ground_truths_tensor = torch.concat(ground_truths_list, dim=0)
        numerical_ids_tensor = torch.concat(numerical_ids_list, dim=0)
        self._sklearn_metrics(output_tensor, ground_truths_tensor, "test", numerical_ids_tensor)

    def _sklearn_metrics(
        self, output: torch.Tensor, ground_truths: torch.Tensor, mode: str, numerical_ids: torch.Tensor
    ):
        logger.info(("output shape", output.shape))
        logger.info(("ground_truths shape", ground_truths.shape))
        logger.info(("numerical_ids shape", numerical_ids.shape))
        model_dir = self.hyperparameters["model_dir"]

        softmax = nn.Softmax(dim=1)
        preds = softmax(output).argmax(dim=1)
        confis = softmax(output).detach().cpu().numpy()
        y_pred = self.label_encoder.inverse_transform(preds.detach().cpu().numpy())
        y_true = self.label_encoder.inverse_transform(ground_truths.detach().cpu().numpy())
        numerical_id_ = numerical_ids.detach().cpu().numpy()

        report = classification_report(y_true, y_pred, output_dict=True)
        report_confusion_matrix = confusion_matrix(y_true, y_pred, labels=list(self.label_encoder.classes_))

        self.save_reports(model_dir, mode, report_confusion_matrix, report)
        self.save_test_evaluations(model_dir, mode, y_pred, y_true, confis, numerical_id_)

    def save_reports(self, model_dir, mode, report_confusion_matrix, report):
        """Save classification report and confusion matrix to csv file.

        Args:
            model_dir: path
            mode: train, test or val
            report_confusion_matrix: sklearn confusion matrix
            report: sklear classification report

        """
        df_cm = pd.DataFrame(report_confusion_matrix)
        df_cr = pd.DataFrame(report).transpose()
        df_cm.to_csv(f"{model_dir}/{mode}_confusion_matrix.csv", sep=";")
        df_cr.to_csv(f"{model_dir}/{mode}_classification_report.csv", sep=";")
        logger.info("Confusion Matrix and Classification report are saved.")

    def save_test_evaluations(self, model_dir, mode, y_pred, y_true, confis, numerical_id_):
        """Save pandas dataframe with prediction and ground truth and identifier (numerical id) of the test dataset."""
        df_test = pd.DataFrame()
        df_test["pred"] = y_pred
        df_test["confidence"] = confis.max(axis=1)
        df_test["label"] = y_true
        df_test["numerical_id"] = numerical_id_
        df_test.to_csv(f"{model_dir}/{mode}_labels_predictions.csv", sep=";")
        logger.info("The label predictions are saved.")


class LitModel(pl.LightningModule):
    """Lightning model for classification."""

    def __init__(
        self,
        hyperparameters: dict,
    ):
        super().__init__()
        self.hyperparameters = hyperparameters
        EC = EncoderAndTokenizer()
        self.tokenizer = EC.load_tokenizer()
        self.label_encoder = EC.load_labelencoder()
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.num_classes = len(self.label_encoder.classes_)
        self.hyperparameters["num_classes"] = self.num_classes
        self.module = BertResNetClassifier(
            endpoint_mode=self.hyperparameters["endpoint_mode"], hyperparameters=self.hyperparameters
        )
        # Classification
        self.criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection(
            [
                MulticlassAccuracy(self.num_classes),
                MulticlassPrecision(self.num_classes),
                MulticlassRecall(self.num_classes),
                MulticlassF1Score(self.num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.pred_list = []
        self.save_hyperparameters()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Forward path, calculate the computational graph in the forward direction.

        Used for train, test and val.

        Returns:
            computional graph

        """
        return self.module(x, y, z)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict:
        """Call the eval share for training.

        Returns:
            dict with loss, outputs and ground_truth.

        """
        return self._shared_eval_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        """Call the eval share for validation.

        Returns:
            dict with loss, outputs and ground_truth.

        """
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict:
        """Call the eval share for test.

        Returns:
            dict with loss, outputs and ground_truth.

        """
        ret = self._shared_eval_step(batch, "test")
        self.pred_list.append(ret)
        return ret

    def _shared_eval_step(self, batch: dict[str, torch.Tensor], mode: str) -> dict:
        """Calculate the desired metrics.

        Args:
            batch: tensor
            mode: train, test or val

        Returns:
            dict with loss, outputs and ground_truth

        """
        ids = batch["ID"]
        atts = batch["Att"]
        ground_truth = batch["GT"]
        numerical_id = batch["NID"]
        img = batch["IMG"]
        out = self.forward(ids, atts, img)
        if mode == "train":
            loss = self.criterion(out, ground_truth)
            output = self.train_metrics(out, ground_truth)
            self.log_dict(output)
            self.train_metrics.update(out, ground_truth)
        elif mode == "val":
            loss = self.criterion(out, ground_truth)
            output = self.val_metrics(out, ground_truth)
            self.val_metrics.update(out, ground_truth)
        elif mode == "test":
            loss = self.criterion(out, ground_truth)
            output = self.test_metrics(out, ground_truth)
            self.test_metrics.update(out, ground_truth)
            # reset predict list
            self.pred_list = []

        return {"outputs": out, "loss": loss, "ground_truth": ground_truth, "numerical_id": numerical_id}

    def _epoch_end(self, mode: str) -> None:
        """Calculate loss and metrics at end of epoch."""
        if mode == "val":
            output = self.val_metrics.compute()
            self.log_dict(output)
            self.val_metrics.reset()
        if mode == "train":
            output = self.train_metrics.compute()
            self.log_dict(output)
            self.train_metrics.reset()
        if mode == "test":
            output = self.test_metrics.compute()
            self.log_dict(output)
            self.test_metrics.reset()

    def predict(self, batch: dict[str, torch.Tensor], batch_idx: int = 0, dataloader_idx: int = 0) -> torch.Tensor:
        """Model prediction  without softmax and argmax to predict class label."""
        self.eval()
        with torch.no_grad():
            ids = batch["ID"]
            atts = batch["Att"]
            img = batch["IMG"]
            return self.forward(ids, atts, img)

    def on_test_epoch_end(self) -> None:
        """Calculate the metrics at the end of epoch for test step."""
        self._epoch_end("test")

    def on_validation_epoch_end(self) -> None:
        """Calculate the metrics at the end of epoch for val step."""
        self._epoch_end("val")

    def on_train_epoch_end(self) -> None:
        """Calculate the metrics at the end of epoch for train step."""
        self._epoch_end("train")

    def configure_optimizers(self) -> Any:
        """Configure the optimizer.

        Returns:
            optimizer

        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.hyperparameters["weight_decay"])
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def configure_callbacks(self) -> Union[Sequence[pl.pytorch.Callback], pl.pytorch.Callback]:
        """Configure Early stopping or Model Checkpointing."""
        early_stop = EarlyStopping(
            monitor="val_MulticlassAccuracy", patience=self.hyperparameters["patience"], mode="max"
        )
        checkpoint = ModelCheckpoint(
            monitor="val_MulticlassAccuracy",
            mode="max",
            dirpath=self.hyperparameters["model_dir"] + "/",
            filename=self.hyperparameters["model_filename"],
        )

        sklearn = SklearnMetricsCallback(label_encoder=self.label_encoder, hyperparameters=self.hyperparameters)
        return [early_stop, checkpoint, sklearn]
