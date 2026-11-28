"""Guards the test harness itself: what a test puts in the environment must reach its dataloader workers."""

import os

import pytest
from torch.utils.data import DataLoader, Dataset

_ENV_VAR = "LITDATA_TEST_WORKER_ENVIRONMENT"


class _EnvDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return os.environ.get(_ENV_VAR, "")


# Two values, so that a worker started from a process another test left behind reports the wrong one.
@pytest.mark.parametrize("value", ["first", "second"])
def test_dataloader_workers_see_environment_set_by_the_test(value, monkeypatch):
    monkeypatch.setenv(_ENV_VAR, value)

    seen = [item for batch in DataLoader(_EnvDataset(), num_workers=2) for item in batch]

    assert seen == [value, value]
