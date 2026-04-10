import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# TODO: Import your actual Dataset and Model
# from dataset import AlzeyeDataModule
# from model import Model 

class LitClassifier(L.LightningModule):
    def __init__(self, lr=2e-4, weight_decay=1e-4, img_size=224):
        super().__init__()
        self.save_hyperparameters()

        # TODO: Initialize your model here
        # self.model = Model(img_size=img_size)
        
        # TODO: Define your loss function here
        # self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # TODO: Define the forward pass
        # return self.model(x)
        pass

    def training_step(self, batch, batch_idx):
        # TODO: Extract data from batch, pass through model, and calculate loss
        # x, y = batch['image'], batch['label']
        # preds = self(x)
        # loss = self.loss_fn(preds, y)
        
        # self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        # return loss
        pass

    def validation_step(self, batch, batch_idx):
        # TODO: Implement validation step
        # x, y = batch['image'], batch['label']
        # preds = self(x)
        # loss = self.loss_fn(preds, y)
        
        # self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        pass

    def test_step(self, batch, batch_idx):
        # TODO: Implement test step
        # x, y = batch['image'], batch['label']
        # preds = self(x)
        # loss = self.loss_fn(preds, y)
        
        # self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        pass

    def configure_optimizers(self):
        # TODO: Define optimizer and (optionally) learning rate scheduler
        # optimizer = torch.optim.AdamW(
        #     self.parameters(), 
        #     lr=self.hparams.lr, 
        #     weight_decay=self.hparams.weight_decay
        # )
        # return optimizer
        pass


def main():
    parser = argparse.ArgumentParser(description="Minimal PyTorch Lightning Training Script")

    # Data & Model args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Trainer args
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed everything for reproducibility
    L.seed_everything(args.seed, workers=True)

    # TODO: Initialize DataModule
    # datamodule = AlzeyeDataModule(
    #     batch_size=args.batch_size,
    #     img_size=args.img_size,
    #     num_workers=args.num_workers,
    # )

    # Initialize LightningModule
    model = LitClassifier(
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # Fit & evaluate
    # TODO: Pass the datamodule once implemented
    # trainer.fit(model, datamodule=datamodule)
    # trainer.test(ckpt_path="best", datamodule=datamodule)

if __name__ == "__main__":
    main()