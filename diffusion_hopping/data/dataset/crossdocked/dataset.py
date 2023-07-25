import os
import random
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch_geometric.data import extract_tar

from diffusion_hopping.data.dataset.crossdocked.provider import CrossDockedProvider
from diffusion_hopping.data.dataset.dataset import ProteinLigandDataset
from diffusion_hopping.data.util import slugify


class CrossDockedDataset(ProteinLigandDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        seed: int = 1,
    ):
        self.seed = seed
        self.randomised_train_test_val_split = None
        super().__init__(root, split, transform, pre_transform, pre_filter, log=log)

    def download(self):
        if all(
            [
                (Path(self.processed_dir) / file_name).exists()
                for file_name in self.processed_file_names
            ]
        ):
            self._log("Not downloading as all processed files are present")
            return
        raw_dir = Path(self.raw_dir)
        if (
            not (raw_dir / "crossdocked_pocket10.tar.gz").exists()
            and not (raw_dir / "crossdocked_pocket10").exists()
        ):
            raise RuntimeError(
                f"Please place the crossdocked_pocket10.tar.gz in the raw directory '{raw_dir}'"
            )
        elif not (raw_dir / "crossdocked_pocket10").exists():
            extract_tar(raw_dir / "crossdocked_pocket10.tar.gz", raw_dir)
            os.remove(raw_dir / "crossdocked_pocket10.tar.gz")

        if not (raw_dir / "split_by_name.pt").exists():
            raise RuntimeError(
                f"Please place the split_by_name.pt in the raw directory '{raw_dir}'"
            )

    def process(self):
        self.randomised_train_test_val_split = self._load_splits(self.seed)
        super().process()

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "crossdocked_pocket10",
            "split_by_name.pt",
        ]

    def _load_splits(self, seed: int = 1):
        rng = random.Random(seed)
        raw_splits = torch.load(Path(self.raw_dir) / "split_by_name.pt")
        all_train_candidates = raw_splits["train"]
        rng.shuffle(all_train_candidates)
        raw_splits["train"] = all_train_candidates[:-300]
        raw_splits["val"] = all_train_candidates[-300:]
        splits = {
            split: {
                slugify(protein_path): (protein_path, ligand_path)
                for protein_path, ligand_path in items
            }
            for split, items in raw_splits.items()
        }
        return splits

    def _get_provider(self) -> CrossDockedProvider:
        return CrossDockedProvider(
            Path(self.raw_dir) / "crossdocked_pocket10",
            split_object=self.randomised_train_test_val_split,
        )

    def _get_split_candidates(self, split: str) -> List[str]:
        return list(self.randomised_train_test_val_split[split].keys())
