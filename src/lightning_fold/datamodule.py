import logging
from abc import ABC, abstractmethod
from typing import Optional

from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)


class FoldDataModule(LightningDataModule, ABC):
    def __init__(self):
        super().__init__()

        self.current_fold_index = -1

    def setup(self, stage: Optional[str] = None) -> None:
        # setup the first fold
        assert self.current_fold_index == -1, self.current_fold_index

        self.current_fold_index = 0
        self.setup_fold(self.current_fold_index)

    def on_advance_start(self):
        """
        This method will be called by the ``FoldLoop``
        :return:
        """
        if self.current_fold_index > 0:  # fold zero is setup in the ``setup`` method
            self.setup_fold(self.current_fold_index)
        logger.info(f"Current fold: {self.current_fold_index}")

    def on_advance_end(self):
        self.teardown_fold(self.current_fold_index)
        self.current_fold_index += 1

    @abstractmethod
    def done(self) -> bool:
        """
        This method will be called by the ``FoldLoop.done``
        :return:
        """

    @abstractmethod
    def setup_fold(self, fold_index: int):
        """
        The used should override this function
        :param fold_index:
        :return:
        """

    @abstractmethod
    def teardown_fold(self, fold_index: int):
        """
        :param fold_index:
        :return:
        """
