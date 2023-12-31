import torch

from .. import builder

from pytorch_lightning.core import LightningModule


class PretrainModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        # load checkpoint
        if self.cfg.train.load_ckpt is not None:
            self.afloc = builder.build_model_from_ckpt(self.cfg.train.load_ckpt)
        else:
            self.afloc = builder.build_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None
        self.trainer = None
        

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.afloc)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, "train")
        loss = res["loss"]

        return loss

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, "val")
        loss = res["loss"]
        return loss

    def test_step(self, batch, batch_idx):
        res = self.shared_step(batch, "test")

    def shared_step(self, batch, split):
        """Similar to traning step"""
        output = self.afloc(batch)

        loss = self.afloc.calc_loss(output, batch)

        # log training progress
        log_iter_loss = True if split == "train" else False
        for k, v in loss.items():
            if 'loss' in k:
                self.log(
                    f"{split}_{k}",
                    v,
                    on_epoch=True,
                    on_step=log_iter_loss,
                    logger=True,
                    prog_bar=False,
                )

        return loss