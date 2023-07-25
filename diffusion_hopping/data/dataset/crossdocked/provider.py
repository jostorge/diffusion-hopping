from pathlib import Path
from typing import Dict, Iterator, Tuple

from torch_geometric.transforms import Compose

from diffusion_hopping.data.dataset.provider import Provider
from diffusion_hopping.data.protein_ligand import Ligand, Protein
from diffusion_hopping.data.transform.obabel import ObabelTransform
from diffusion_hopping.data.transform.reduce import ReduceTransform


class CrossDockedProvider(Provider):
    def __init__(self, path: Path, split_object: Dict[str, Dict[str, Tuple[str, str]]]):
        super().__init__()
        self.path = path
        self.path_mapping = {
            key: (protein_path, ligand_path)
            for split, paths in split_object.items()
            for key, (protein_path, ligand_path) in paths.items()
        }
        assert len(self.path_mapping) == sum(
            len(paths) for paths in split_object.values()
        ), "Duplicate keys in split object"
        self.protein_transform = Compose([ObabelTransform(), ReduceTransform()])
        self.ligand_transform = ObabelTransform(from_format="sdf", to_format="sdf")

    def __iter__(self) -> Iterator[str]:
        return iter(self.path_mapping)

    def process_ligand(self, index) -> Ligand:
        ligand_path = self.path / self.path_mapping[index][1]
        transform = self.ligand_transform
        return Ligand(transform(ligand_path))

    def process_protein(self, index) -> Protein:
        protein_path = self.path / self.path_mapping[index][0]
        transform = self.protein_transform
        return Protein(transform(protein_path))
