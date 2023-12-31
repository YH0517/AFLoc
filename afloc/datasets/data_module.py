import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from . import pretraining_dataset
from .. import builder


class PretrainingDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dataset = pretraining_dataset.MultimodalPretrainingDataset
        self.collate_fn = pretraining_dataset.multimodal_collate_fn
        if self.cfg.lightning.trainer.accelerator == "ddp":
            self.batch_size = self.cfg.train.per_gpu_batchsize
        else:
            self.batch_size = self.cfg.train.batch_size
        self.pin_memory = False
        print("batch size: ", self.batch_size)

    def train_dataloader(self):
        transform = builder.build_transformation(self.cfg, "train")
        dataset = self.dataset(self.cfg, split="train", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="validate", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )

    def test_dataloader(self):
        transform = builder.build_transformation(self.cfg, "test")
        dataset = self.dataset(self.cfg, split="test", transform=transform)
        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.cfg.train.num_workers,
        )
    
