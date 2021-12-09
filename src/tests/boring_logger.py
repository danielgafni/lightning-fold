from typing import Dict, Optional

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only


class BoringLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.hparams_logged = None
        self.metrics_logged: Dict[int, Dict[str, float]] = {}
        self.finalized = False
        self.after_save_checkpoint_called = False

    @property
    def experiment(self):
        return "test"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hparams_logged = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.metrics_logged[step] = metrics

    @rank_zero_only
    def finalize(self, status):
        self.finalized = status

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None

    @property
    def name(self):
        return "name"

    @property
    def version(self):
        return "1"

    def after_save_checkpoint(self, checkpoint_callback):
        self.after_save_checkpoint_called = True
