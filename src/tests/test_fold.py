from itertools import product

import pytest
from pytorch_lightning import Trainer, seed_everything

from lightning_fold import FoldLoop
from tests.boring_fold_datamodule import BoringFoldDataModule
from tests.boring_logger import BoringLogger
from tests.boring_model import BoringModel


@pytest.mark.parametrize("num_folds, dataset_length, max_epochs", product((1, 2, 4), (5, 10), (1, 2)))
def test_fold(num_folds: int, dataset_length: int, max_epochs: int):
    seed_everything(0)

    in_size = 2

    model = BoringModel(in_size=in_size)
    datamodule = BoringFoldDataModule(length=dataset_length, width=in_size, num_folds=num_folds)

    logger = BoringLogger()
    trainer = Trainer(max_epochs=max_epochs, logger=logger, enable_checkpointing=False)
    trainer.fit_loop = FoldLoop(trainer.fit_loop)  # type: ignore

    trainer.fit(model=model, datamodule=datamodule)

    assert set(datamodule.passed_folds) == set(range(num_folds))

    total_steps = num_folds * dataset_length * max_epochs

    assert total_steps == sum(datamodule.passed_train_steps)
    assert total_steps == len(logger.metrics_logged)
