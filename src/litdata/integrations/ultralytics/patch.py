# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from litdata.constants import _ULTRALYTICS_AVAILABLE
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset


def patch_ultralytics() -> None:
    """Patch Ultralytics to use the LitData optimize function."""
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is not available. Please install it to use this functionality.")

    from ultralytics.data.utils import check_det_dataset

    check_det_dataset.__code__ = patch_check_det_dataset.__code__

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    DetectionTrainer.plot_training_samples = patch_detection_plot_training_samples
    DetectionTrainer.plot_training_labels = patch_none_function

    from ultralytics.models.yolo.detect.val import DetectionValidator

    DetectionValidator.plot_val_samples = patch_detection_plot_val_samples
    DetectionValidator.plot_predictions = patch_detection_plot_predictions

    import ultralytics.data.base as base_module
    import ultralytics.data.dataset as child_modules

    base_module.BaseDataset = PatchedUltralyticsBaseDataset
    base_module.BaseDataset.set_rectangle = patch_none_function
    child_modules.YOLODataset.__bases__ = (PatchedUltralyticsBaseDataset,)
    child_modules.YOLODataset.get_labels = patch_get_labels

    from ultralytics.data.build import build_dataloader

    build_dataloader.__code__ = patch_build_dataloader.__code__

    from ultralytics.data.augment import Compose

    Compose.__call__.__code__ = patch_compose_transform_call.__code__

    print("✅ Ultralytics successfully patched to use LitData.")


if _ULTRALYTICS_AVAILABLE:
    from ultralytics.data.base import BaseDataset as UltralyticsBaseDataset
    from ultralytics.utils.plotting import plot_images

    class PatchedUltralyticsBaseDataset(UltralyticsBaseDataset):
        def __init__(self: Any, img_path: str, classes: Optional[List[int]] = None, *args: Any, **kwargs: Any):
            print("patched ultralytics dataset: 🔥")
            self.litdata_dataset = img_path
            self.classes = classes
            super().__init__(img_path, classes=classes, *args, **kwargs)
            self.streaming_dataset = StreamingDataset(
                img_path,
                transform=[
                    ultralytics_detection_transform,
                    partial(self.transform_update_label, classes),
                    self.update_labels_info,
                    self.transforms,
                ],
                transform_kwargs={
                    "img_size": self.imgsz,
                    "channels": self.channels,
                    "segment": self.use_segments,
                    "use_keypoints": self.use_keypoints,
                    "use_obb": self.use_obb,
                    "lit_args": self.data,
                    "single_cls": self.single_cls,
                },
            )
            self.ni = len(self.streaming_dataset)
            self.buffer = list(range(len(self.streaming_dataset)))

        def __len__(self: Any) -> int:
            """Return the length of the dataset."""
            return len(self.streaming_dataset)

        def get_image_and_label(self: Any, index: int) -> None:
            # Your custom logic to load from .litdata
            # e.g. use `self.litdata_dataset[index]`
            raise NotImplementedError("Custom logic here")

        def get_img_files(self: Any, img_path: Union[str, List[str]]) -> List[str]:
            """Let this method return an empty list to avoid errors."""
            return []

        def get_labels(self: Any) -> List[Dict[str, Any]]:
            # this is used to get number of images (ni) in the BaseDataset class
            return []

        def cache_images(self: Any) -> None:
            pass

        def cache_images_to_disk(self: Any, i: int) -> None:
            pass

        def check_cache_disk(self: Any, safety_margin: float = 0.5) -> bool:
            """Check if the cache disk is available."""
            # This method is not used in the streaming dataset, so we can return True
            return True

        def check_cache_ram(self: Any, safety_margin: float = 0.5) -> bool:
            """Check if the cache RAM is available."""
            # This method is not used in the streaming dataset, so we can return True
            return True

        def update_labels(self: Any, *args: Any, **kwargs: Any) -> None:
            """Do nothing, we will update labels when item is fetched in transform."""
            pass

        def transform_update_label(
            self: Any, include_class: Optional[List[int]], label: Dict, *args: Any, **kwargs: Any
        ) -> Dict:
            """Update labels to include only specified classes.

            Args:
                self: PatchedUltralyticsBaseDataset instance.
                include_class (List[int], optional): List of classes to include. If None, all classes are included.
                label (Dict): Label to update.
                *args: Additional positional arguments (unused).
                **kwargs: Additional keyword arguments (unused).
            """
            include_class_array = np.array(include_class).reshape(1, -1)
            if include_class is not None:
                cls = label["cls"]
                bboxes = label["bboxes"]
                segments = label["segments"]
                keypoints = label["keypoints"]
                j = (cls == include_class_array).any(1)
                label["cls"] = cls[j]
                label["bboxes"] = bboxes[j]
                if segments:
                    label["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    label["keypoints"] = keypoints[j]
            if self.single_cls:
                label["cls"][:, 0] = 0

            return label

        def __getitem__(self: Any, index: int) -> Dict[str, Any]:
            """Return transformed label information for given index."""
            # return self.transforms(self.get_image_and_label(index))
            if not hasattr(self, "streaming_dataset"):
                raise ValueError("The dataset must have a 'streaming_dataset' attribute.")
            data = self.streaming_dataset[index]

            label = data["label"]
            # split label on the basis of `\n` and then split each line on the basis of ` `
            # first element is class, rest are bbox coordinates
            if isinstance(label, str):
                label = label.split("\n")
                label = [line.split(" ") for line in label if line.strip()]

                data = {
                    "batch_idx": torch.tensor([index], dtype=torch.int32),
                    "img": data["image"],
                    "cls": torch.Tensor([int(line[0]) for line in label]),
                    "bboxes": torch.Tensor([[float(coord) for coord in line[1:]] for line in label]),
                    "normalized": True,
                    "segments": [],
                    "keypoints": None,
                    "bbox_format": "xywh",
                }
            else:
                raise ValueError("Label must be a string in YOLO format.")

            data = self.transform_update_label(
                include_class=self.classes,
                label=data,
            )
            return self.transforms(data)

    def patch_detection_plot_training_samples(self: Any, batch: Dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            self: DetectionTrainer instance.
            batch (Dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            images=batch["img"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def patch_detection_plot_val_samples(self: Any, batch: Dict[str, Any], ni: int) -> None:
        """Plot validation image samples.

        Args:
            self: DetectionValidator instance.
            batch (Dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            images=batch["img"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def patch_detection_plot_predictions(
        self: Any, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int, max_det: Optional[int] = None
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            self: DetectionValidator instance.
            batch (Dict[str, Any]): Batch containing images and annotations.
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        from ultralytics.utils import ops

        # TODO: optimize this
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        # TODO: fix this
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=None,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def patch_check_det_dataset(dataset: str, _: bool = True) -> Dict:
        if not (isinstance(dataset, str) and dataset.endswith(".yaml") and os.path.isfile(dataset)):
            raise ValueError("Dataset must be a string ending with '.yaml' and point to a valid file.")

        import yaml

        if not dataset.startswith("litdata_"):
            dataset = "litdata_" + dataset

        if not os.path.isfile(dataset):
            raise FileNotFoundError(f"Dataset file not found: {dataset}")

        # read the yaml file
        with open(dataset) as file:
            return yaml.safe_load(file)

    def patch_build_dataloader(
        dataset: Any, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False
    ) -> StreamingDataLoader:
        """Create and return an InfiniteDataLoader or DataLoader for training or validation.

        Args:
            dataset (Dataset): Dataset to load data from.
            batch (int): Batch size for the dataloader.
            workers (int): Number of worker threads for loading data.
            shuffle (bool, optional): Whether to shuffle the dataset.
            rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
            drop_last (bool, optional): Whether to drop the last incomplete batch.

        Returns:
            (StreamingDataLoader): A dataloader that can be used for training or validation.

        Examples:
            Create a dataloader for training
            >>> dataset = YOLODataset(...)
            >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
        """
        from litdata.streaming.dataloader import StreamingDataLoader

        print("litdata is rocking⚡️")
        if not hasattr(dataset, "streaming_dataset"):
            raise ValueError("The dataset must have a 'streaming_dataset' attribute.")

        from ultralytics.data.utils import PIN_MEMORY

        batch = min(batch, len(dataset))
        num_devices = torch.cuda.device_count()  # number of CUDA devices
        num_workers = min((os.cpu_count() or 1) // max(num_devices, 1), workers)  # number of workers
        persistent_workers = bool(int(os.getenv("UL_PERSISTENT_WORKERS", 0)))
        return StreamingDataLoader(
            dataset=dataset.streaming_dataset,
            batch_size=batch,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=PIN_MEMORY,
            collate_fn=getattr(dataset, "collate_fn", None),
            drop_last=drop_last,
        )

    def patch_detection_plot_training_labels(self: Any) -> None:
        """Create a labeled training plot of the YOLO model."""
        pass

    def patch_get_labels(self: Any) -> List[Dict[str, Any]]:
        # this is used to get number of images (ni) in the BaseDataset class
        return []

    def patch_none_function(*args: Any, **kwargs: Any) -> None:
        """A placeholder function that does nothing."""
        pass

    def image_resize(
        im: Any, imgsz: int, rect_mode: bool = True, augment: bool = True
    ) -> Tuple[Any, Tuple[int, int], Tuple[int, int]]:
        """Resize the image to a fixed size.

        Args:
            im (Any): Image to resize.
            imgsz (int): Target size for resizing.
            rect_mode (bool): Whether to use rectangle mode for resizing.
            augment (bool): If True, data augmentation is applied.

        Returns:
            Tuple[Any, Tuple[int, int], Tuple[int, int]]: Resized image and its original dimensions.
        """
        import cv2

        # Custom logic for resizing the image
        h0, w0 = im.shape[:2]  # orig hw
        if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            r = imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        if im.ndim == 2:
            im = im[..., None]

        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def patch_compose_transform_call(self: Any, data: Any) -> Any:
        """Apply all transforms to the data, skipping mix transforms."""
        from ultralytics.data.augment import BaseMixTransform

        for t in self.transforms:
            if isinstance(t, BaseMixTransform):
                continue  # Skip mix transforms, they are applied separately
            data = t(data)
        return data


# ------- helper transformations -------


def ultralytics_detection_transform(data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Transform function for YOLO detection datasets.

    Args:
        data (Dict[str, Any]): Input data containing image and label.
        kwargs (Dict[str, Any]): Additional keyword arguments, including the index of the data item.

    Returns:
        Dict[str, Any]: Transformed data with image and label.
    """
    index = kwargs.get("index")
    channels = kwargs.get("channels", 3)  # default to 3 channels (RGB)
    if index is None:
        raise ValueError("Index must be provided for YOLO detection transform.")

    label = data["label"]
    # split label on the basis of `\n` and then split each line on the basis of ` `
    # first element is class, rest are bbox coordinates
    if isinstance(label, str):
        img, ori_shape, resized_shape = image_resize(
            data["img"], imgsz=kwargs.get("img_size", 640), rect_mode=True, augment=True
        )
        ratio_pad = (
            resized_shape[0] / ori_shape[0],
            resized_shape[1] / ori_shape[1],
        )  # for evaluation
        lb, segments, keypoint = parse_labels(label, **kwargs)

        data = {
            "batch_idx": np.array([index]),  # ← add this!
            "img": img,
            "cls": lb[:, 0:1],  # n, 1
            "bboxes": lb[:, 1:],  # n, 4
            "segments": segments,
            "keypoints": keypoint,
            "normalized": True,
            "bbox_format": "xywh",
            "ori_shape": ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
            "channels": channels,
        }
    else:
        raise ValueError("Label must be a string in YOLO format.")

    return data


def parse_labels(labels: str, **kwargs: Any) -> Tuple[Any, Any, Any]:
    from ultralytics.utils.ops import segments2boxes

    keypoint = kwargs.get("keypoint", False)
    single_cls = kwargs.get("single_cls", False)
    data = kwargs["lit_args"]
    nkpt, ndim = data.get("kpt_shape", (0, 0))
    num_cls: int = len(data["names"])

    segments: Any = []
    keypoints: Any = None

    lb = [x.split() for x in labels.split("\n") if len(x)]
    if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
        classes = np.array([x[0] for x in lb], dtype=np.float32)
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
    lb = np.array(lb, dtype=np.float32)
    if nl := len(lb):
        if keypoint:
            assert lb.shape[1] == (5 + nkpt * ndim), (
                f"labels require {(5 + nkpt * ndim)} columns each, but {lb.shape[1]} columns detected"
            )
            points = lb[:, 5:].reshape(-1, ndim)[:, :2]
        else:
            assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
            points = lb[:, 1:]
        # Coordinate points check with 1% tolerance
        assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
        assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"

        # All labels
        if single_cls:
            lb[:, 0] = 0
        max_cls = lb[:, 0].max()  # max label count
        assert max_cls < num_cls, (
            f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
            f"Possible class labels are 0-{num_cls - 1}"
        )
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
            lb = lb[i]  # remove duplicates
            if segments:
                segments = [segments[x] for x in i]
    if keypoint:
        keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
        if ndim == 2:
            kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
    lb = lb[:, :5]
    return lb, segments, keypoints
