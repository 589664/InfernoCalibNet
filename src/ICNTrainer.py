import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from config import NUM_CL


class ICNTrainer(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Metrics
        self.auroc = MultilabelAUROC(num_labels=NUM_CL, average="macro")
        self.f1_score = MultilabelF1Score(num_labels=NUM_CL, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        if targets.sum() > 0:  # Skip metric updates if no positive samples
            self.auroc.update(outputs, targets.int())
            self.f1_score.update(outputs, targets.int())
            self.log("train_auroc", self.auroc.compute(), on_epoch=True, logger=True)
            self.log("train_f1", self.f1_score.compute(), on_epoch=True, logger=True)
        else:
            self.log("train_auroc", float("nan"), on_epoch=True, logger=True)
            self.log("train_f1", float("nan"), on_epoch=True, logger=True)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.auroc.reset()
        self.f1_score.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        if targets.sum() > 0:  # Skip metric updates if no positive samples
            self.auroc.update(outputs, targets.int())
            self.f1_score.update(outputs, targets.int())
            self.log("val_auroc", self.auroc.compute(), on_epoch=True, logger=True)
            self.log("val_f1", self.f1_score.compute(), on_epoch=True, logger=True)
        else:
            self.log("val_auroc", float("nan"), on_epoch=True, logger=True)
            self.log("val_f1", float("nan"), on_epoch=True, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.auroc.reset()
        self.f1_score.reset()

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
            },
        }
