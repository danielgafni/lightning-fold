import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class BoringModel(LightningModule):
    def __init__(self, in_size: int):
        """Testing PL Module.
        Use as follows:
        - subclass
        - modify the behavior for what you want
        class TestModel(BaseTestModel):
            def training_step(...):
                # do your own thing
        or:
        model = BaseTestModel()
        model.training_epoch_end = None
        """
        super().__init__()
        self.layer = torch.nn.Linear(in_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def loss(self, output):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return self.criterion(output, torch.ones_like(output))

    def step(self, batch):
        x = batch[0]
        return self(x)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        loss = self.loss(output)
        self.log("Train/loss", loss)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        loss = self.loss(output)
        self.log("Val/loss", loss)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["x"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.step(batch)
        loss = self.loss(output)
        self.log("Test/loss", loss)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
