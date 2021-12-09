from typing import List, Optional

import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lightning_fold import FoldDataModule


class BoringFoldDataModule(FoldDataModule):
    def __init__(self, length: int = 10, width: int = 5, num_folds: int = 4):
        super().__init__()

        self.length = length
        self.width = width
        self.num_folds = num_folds

        self.train_fold: Optional[Dataset] = None
        self.val_fold: Optional[Dataset] = None
        self.test_fold: Optional[Dataset] = None

        self.passed_folds: List[int] = []
        self.passed_train_steps: List[int] = []
        self.passed_val_steps: List[int] = []
        self.passed_test_steps: List[int] = []

    def setup_fold(self, fold_index: int):
        """
        Emulate creating new folds of data
        """
        self.train_fold = TensorDataset(torch.rand(self.length, self.width))
        self.val_fold = TensorDataset(torch.rand(self.length, self.width))
        self.test_fold = TensorDataset(torch.rand(self.length, self.width))

    def teardown_fold(self, fold_index: int):
        self.passed_folds.append(fold_index)
        self.passed_train_steps.append(len(self.train_dataloader()))
        self.passed_val_steps.append(len(self.val_dataloader()))
        self.passed_test_steps.append(len(self.test_dataloader()))

    def done(self) -> bool:
        return self.current_fold_index >= self.num_folds  # type: ignore

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train_fold is not None

        return DataLoader(self.train_fold)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.val_fold is not None

        return DataLoader(self.val_fold)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_fold is not None

        return DataLoader(self.test_fold)
