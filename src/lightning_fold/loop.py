import logging
from os.path import join
from typing import Any, Dict, Optional

from pytorch_lightning.loops import FitLoop, Loop
from pytorch_lightning.trainer.states import TrainerFn

from lightning_fold import FoldDataModule

logger = logging.getLogger()


class FoldLoop(Loop):
    """
    Should be used with the ``FoldDataModule`` to train on multiple folds
    """

    def __init__(self, fit_loop: FitLoop, save_dir: Optional[str] = None):
        """
        :param fit_loop: `pl.Trainer.fit_loop` attribute
        """
        super().__init__()

        self.fit_loop = fit_loop
        self.save_dir = save_dir
        self._first_block_fitted = False

    @property
    def done(self) -> bool:
        assert isinstance(self.trainer.datamodule, FoldDataModule)  # type: ignore

        return self.trainer.datamodule.done()  # type: ignore

    def reset(self) -> None:
        assert isinstance(self.trainer.datamodule, FoldDataModule)  # type: ignore

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""

        assert isinstance(self.trainer.datamodule, FoldDataModule)  # type: ignore

        assert self.trainer.datamodule.current_fold_index == 0, self.trainer.datamodule.current_fold_index  # type: ignore

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        assert isinstance(self.trainer.datamodule, FoldDataModule)  # type: ignore

        self.trainer.datamodule.on_advance_start()  # type: ignore

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fit loop on the current blocks."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()  # fit loop includes val_loop

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()  # type: ignore

    def on_advance_end(self) -> None:
        # save model checkpoint for future testing
        if self.save_dir is not None:
            test_model_save_path = join(self.save_dir, f"fold={self.trainer.datamodule.current_fold_index}.model.pt")  # type: ignore
            logger.info(f"Saving model to {test_model_save_path}")
            self.trainer.save_checkpoint(test_model_save_path)  # type: ignore

        self.trainer.datamodule.on_advance_end()  # type: ignore

        # self.global_step += self.fit_loop.global_step
        # self.current_epoch += self.fit_loop.current_epoch
        # self.total_batch_idx += self.fit_loop.batch_idx

    def on_run_end(self) -> None:
        pass

    def on_save_checkpoint(self) -> Dict[str, int]:
        pass
        # return {"current_fold_index": self.current_fold_index}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        pass
        # self.current_fold_index = state_dict["current_fold_index"]

    # the code below is copied from pytorch-lightning's custom loop example:
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_val_dataloaders(self.trainer.model)  # type: ignore
        self.trainer.state.fn = TrainerFn.FITTING  # type: ignore
        self.trainer.training = True  # type: ignore
        self.fit_loop.current_epoch = 0

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader(self.trainer.model)  # type: ignore
        self.trainer.state.fn = TrainerFn.TESTING  # type: ignore
        self.trainer.testing = True  # type: ignore

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
