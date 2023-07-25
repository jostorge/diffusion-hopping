import os
from abc import abstractmethod

from pytorch_lightning import LightningDataModule
from rdkit import Chem
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class ProteinLigandComplexDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        test_batch_size=None,
        val_batch_size=None,
        shuffle=True,
        overfit_item=False,
    ) -> None:
        super().__init__()
        self.predict_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size if test_batch_size else batch_size
        self.val_batch_size = val_batch_size if val_batch_size else test_batch_size
        self.shuffle = shuffle
        self.overfit_item = overfit_item

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self.dataset_from_split("train")
            self.val_dataset = self.dataset_from_split("val")
        elif stage == "test":
            self.test_dataset = self.dataset_from_split("test")
        elif stage == "predict":
            self.predict_dataset = self.dataset_from_split("predict")
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def dataloader(dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs, num_workers=0)

    @abstractmethod
    def dataset_from_split(self, split: str) -> Dataset:
        ...

    def train_dataloader(self):
        if self.overfit_item:
            train_dataset = [
                self.train_dataset[0] for _ in range(len(self.train_dataset))
            ]
        else:
            train_dataset = self.train_dataset
        return self.dataloader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.overfit_item:
            val_dataset = [self.train_dataset[0] for _ in range(self.batch_size)]
        else:
            val_dataset = self.val_dataset
        return self.dataloader(
            val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.dataloader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.dataloader(
            self.predict_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def get_train_smiles(self):
        if self.train_dataset is None:
            self.setup("fit")
        if self.overfit_item:
            return [Chem.MolToSmiles(self.train_dataset[0].ligand.ref)]
        return [Chem.MolToSmiles(item["ligand"].ref) for item in self.train_dataset]
