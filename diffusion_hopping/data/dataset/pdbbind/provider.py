from pathlib import Path
from typing import Iterator, List, Tuple

from torch_geometric.transforms import Compose

from diffusion_hopping.data.dataset.provider import Provider
from diffusion_hopping.data.protein_ligand import Ligand, Protein
from diffusion_hopping.data.transform.obabel import ObabelTransform
from diffusion_hopping.data.transform.reduce import ReduceTransform


def _candidates_from_folder(path: Path) -> List[Tuple[str, Path]]:
    return [
        (complex.name, complex)
        for complex in path.iterdir()
        if complex.is_dir() and complex.name != "index" and complex.name != "readme"
    ]


class PDBProvider(Provider):
    def __init__(self, paths: List[Path]):
        super().__init__()
        self.paths = paths
        self.ligand_transform_sdf = ObabelTransform(from_format="sdf", to_format="sdf")
        self.ligand_transform_mol2 = ObabelTransform(
            from_format="mol2", to_format="sdf"
        )
        self.protein_transform = Compose([ObabelTransform(), ReduceTransform()])

        candidates = sum([_candidates_from_folder(path) for path in paths], [])
        self.candidates = {index: path for index, path in candidates}

    def __iter__(self) -> Iterator[str]:
        return iter(self.candidates.keys())

    def process_ligand(self, index) -> Ligand:
        local_path = self.candidates[index]
        ligand_paths = [
            local_path / f"{index}_ligand.sdf",
            local_path / f"{index}_ligand.mol2",
        ]
        ligand_transforms = [self.ligand_transform_sdf, self.ligand_transform_mol2]
        for ligand_path, ligand_transform in zip(ligand_paths, ligand_transforms):
            if ligand_path.exists():
                try:

                    # try to convert and read the ligand, if it fails, try the next one
                    ligand = Ligand(ligand_transform(ligand_path))
                    ligand.rdkit_mol(sanitize=True)
                    return ligand

                except ValueError as e:
                    continue
        raise FileNotFoundError(f"Could not find parsable ligand for {index}")

    def process_protein(self, index) -> Protein:
        protein_path = self.candidates[index] / f"{index}_protein.pdb"
        return Protein(self.protein_transform(protein_path))
