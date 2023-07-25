import os
from pathlib import Path
from typing import Callable, List, Optional

from torch_geometric.data import download_url, extract_tar

from diffusion_hopping.data.dataset.dataset import ProteinLigandDataset
from diffusion_hopping.data.dataset.pdbbind.provider import PDBProvider
from diffusion_hopping.data.util import keys_from_file


class PDBBindDataset(ProteinLigandDataset):
    datasets_to_download = [
        # [REDACTED, obtain URLs from http://www.pdbbind.org.cn after registering]
    ]
    train_test_splits_to_download = [
        "https://raw.githubusercontent.com/gcorso/DiffDock/724da9406b452686ccd12fef1af8e77d77d31d91/data/splits/timesplit_no_lig_overlap_train",
        "https://raw.githubusercontent.com/gcorso/DiffDock/724da9406b452686ccd12fef1af8e77d77d31d91/data/splits/timesplit_no_lig_overlap_val",
        "https://raw.githubusercontent.com/gcorso/DiffDock/724da9406b452686ccd12fef1af8e77d77d31d91/data/splits/timesplit_test",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
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
        # downloads and extracts the PDBBind datasets
        for url in self.datasets_to_download:
            file_path = download_url(url, self.raw_dir, log=self.log)
            extract_tar(file_path, self.raw_dir)
            os.remove(file_path)

        for file in self.train_test_splits_to_download:
            download_url(file, self.raw_dir, log=self.log)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "v2020-other-PL",
            "refined-set",
            "timesplit_no_lig_overlap_train",
            "timesplit_no_lig_overlap_val",
            "timesplit_test",
        ]

    def _get_provider(self) -> PDBProvider:
        return PDBProvider(
            [
                Path(self.raw_dir) / "v2020-other-PL",
                Path(self.raw_dir) / "refined-set",
            ]
        )

    def _get_split_candidates(self, split: str) -> List[str]:
        if split == "train":
            candidates = keys_from_file(
                Path(self.raw_dir) / "timesplit_no_lig_overlap_train"
            )
        elif split == "val":
            candidates = keys_from_file(
                Path(self.raw_dir) / "timesplit_no_lig_overlap_val"
            )
        elif split == "test":
            candidates = keys_from_file(Path(self.raw_dir) / "timesplit_test")
        else:
            raise ValueError(f"Unknown split: {split}")
        return candidates
