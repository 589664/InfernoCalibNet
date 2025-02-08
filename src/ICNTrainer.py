import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from config import NUM_CL, DISEASE_LABELS
from rich import print


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
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        if targets.sum() > 0:  # Skip metric updates if no positive samples
            self.auroc.update(outputs, targets.int())
            self.f1_score.update(outputs, targets.int())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        # Check if AUROC and F1 metrics have valid data
        try:
            val_auroc = self.auroc.compute()
        except ValueError:
            val_auroc = float("nan")
        self.log("val_auroc", val_auroc, on_epoch=True, logger=True)

        try:
            val_f1 = self.f1_score.compute()
        except ValueError:
            val_f1 = float("nan")
        self.log("val_f1", val_f1, on_epoch=True, logger=True)

        # Reset metrics for the next epoch
        self.auroc.reset()
        self.f1_score.reset()

    def test_step(self, batch, batch_idx):
        # Ensure metrics are on the correct device
        self.auroc = self.auroc.to(self.device)
        self.f1_score = self.f1_score.to(self.device)

        # Reconfigure metrics for per-label computation
        self.auroc = MultilabelAUROC(num_labels=NUM_CL, average=None).to(self.device)
        self.f1_score = MultilabelF1Score(num_labels=NUM_CL, average=None).to(
            self.device
        )

        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        # Update metrics for per-label evaluation
        self.auroc.update(outputs, targets.int())
        self.f1_score.update(outputs, targets.int())

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        # Compute per-label metrics
        test_auroc = self.auroc.compute()
        test_f1 = self.f1_score.compute()

        print(test_auroc)
        print(test_f1)

        # Log and print metrics for each label
        # for idx, disease in enumerate(DISEASE_LABELS):
        #     print(
        #         f"{disease}: AUROC: {test_auroc[idx].item():.4f}, F1: {test_f1[idx].item():.4f}"
        #     )

        # Reset metrics
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
