from abc import abstractmethod
from typing import Iterable, Iterator, List

from diffusion_hopping.data.protein_ligand import Ligand, Protein, ProteinLigandComplex


class Provider(Iterable):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        ...

    def get_keys(self) -> List[str]:
        return list(iter(self))

    @abstractmethod
    def process_ligand(self, index: str) -> Ligand:
        ...

    @abstractmethod
    def process_protein(self, index: str) -> Protein:
        ...

    def get_identifier(self, index: str) -> str:
        return index

    def get_preprocessed_complex(self, index) -> ProteinLigandComplex:
        return ProteinLigandComplex(
            self.process_protein(index),
            self.process_ligand(index),
            identifier=self.get_identifier(index),
        )
