import shutil
from pathlib import Path
from typing import Optional

from biopandas.pdb import PandasPdb
from rdkit import Chem


class Protein:
    def __init__(self, pdb_path) -> None:
        if not Path(pdb_path).exists():
            raise FileNotFoundError(f"Could not find {pdb_path}")
        self.path = pdb_path

    def pandas_pdb(self) -> PandasPdb:
        return PandasPdb().read_pdb(str(self.path))


class Ligand:
    def __init__(self, sdf_path) -> None:
        if not Path(sdf_path).exists():
            raise FileNotFoundError(f"Could not find {sdf_path}")
        self.path = sdf_path

    def rdkit_mol(self, sanitize=True, removeHs=True):
        mol = next(
            Chem.SDMolSupplier(str(self.path), sanitize=sanitize, removeHs=removeHs)
        )
        if mol is None:
            raise ValueError(f"Could not parse {self.path}")
        return mol


class ProteinLigandComplex:
    def __init__(
        self, protein: Protein, ligand: Ligand, identifier: Optional[str] = None
    ) -> None:
        self.protein = protein
        self.ligand = ligand
        self.identifier = identifier

    def to_file(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.protein.path, path / "protein.pdb")
        shutil.copy(self.ligand.path, path / "ligand.sdf")

    @staticmethod
    def from_file(path: Path, identifier: Optional[str] = None):
        return ProteinLigandComplex(
            Protein(path / "protein.pdb"), Ligand(path / "ligand.sdf"), identifier
        )
