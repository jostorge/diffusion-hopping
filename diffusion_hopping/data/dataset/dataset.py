import sys
from abc import abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from diffusion_hopping.data.dataset.provider import Provider
from diffusion_hopping.data.util import ProcessedComplexStorage


class ProteinLigandDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter, log=log)
        split_file = Path(self.processed_dir) / f"{split}.pt"
        self.data, self.slices = torch.load(split_file)

        self.provider = None
        self.processed_complexes: Optional[ProcessedComplexStorage] = None

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.split}.pt"]

    @abstractmethod
    def _get_provider(self) -> Provider:
        ...

    @abstractmethod
    def _get_split_candidates(self, split: str) -> List[str]:
        ...

    def process(self):
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(exist_ok=True, parents=True)

        processed_complexes = ProcessedComplexStorage(
            processed_dir / "processed_complexes"
        )
        self.provider = self._get_provider()
        self.processed_complexes = processed_complexes

        self._preprocess_complexes_into_storage(max_workers=0)
        self._featurize_split()

    def _preprocess_complexes_into_storage(self, max_workers: int = None):
        storage = self.processed_complexes
        keys = self.provider.get_keys()

        if max_workers is None or max_workers == 0:
            did_succeed = []
            for key in tqdm(keys):
                did_succeed.append(self._preprocess_and_store(key))
        else:
            did_succeed = thread_map(
                self._preprocess_and_store, keys, max_workers=max_workers
            )
        num_failed = len(keys) - sum(did_succeed)

        self._log(f"Preprocessed {len(storage)} complexes")
        self._log(f"Failed to preprocess {num_failed} complexes")

    def _preprocess_and_store(self, index: str) -> bool:
        if index not in self.processed_complexes:
            try:
                self.processed_complexes[
                    index
                ] = self.provider.get_preprocessed_complex(index)
                return True
            except Exception as e:
                self._log(f"Could not process {index}: {e}")
                return False
        return True

    def _featurize_split(self, split: str = None):
        if split is None:
            split = self.split

        processed_dir = Path(self.processed_dir)
        split_candidates = self._get_split_candidates(split)

        split_file = processed_dir / f"{split}.pt"

        if split_file.exists():
            self._log(f"Skipping {split} as {split_file} already exists")
            return

        self._log(
            f"Processing {len(split_candidates)} complexes to {split_file} for {split}"
        )
        self._featurize_split_given_candidates(split_candidates, split_file)

    def _featurize_split_given_candidates(
        self, split_candidates: List[str], split_file: Path
    ):
        transformed_complexes = []
        for identifier in (pbar := tqdm(split_candidates)):
            candidate = self._featurize_candidate(identifier)
            if candidate is not None:
                transformed_complexes.append(candidate)
                pbar.set_description(
                    f"Transformed {len(transformed_complexes)} complexes"
                )
        transformed_complexes = [c for c in transformed_complexes if c is not None]
        self._log(f"Transformed {len(transformed_complexes)} complexes")
        data, slices = self.collate(transformed_complexes)
        torch.save((data, slices), split_file)

    def _featurize_candidate(self, identifier: str) -> Optional[Data]:
        try:
            if identifier not in self.processed_complexes:
                return None

            complex = self.processed_complexes[identifier]
            if self.pre_filter is not None and not self.pre_filter(complex):
                return None
            if self.pre_transform is not None:
                complex = self.pre_transform(complex)

            return complex
        except Exception as e:
            self._log(f"Could not process {identifier}: {e}")
            return None

    def _log(self, msg: str):
        if self.log:
            tqdm.write(msg, file=sys.stderr)
