import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data

from diffusion_hopping.data.featurization.util import (
    atomic_symbols_to_one_hot,
    get_ligand_scaffold_mask,
)
from diffusion_hopping.data.protein_ligand import ProteinLigandComplex


class LigandSimpleFeaturization:
    def __init__(self, cutoff=10) -> None:
        self.cutoff = cutoff

    @staticmethod
    def get_ligand_atom_types(ligand: Chem.Mol) -> torch.Tensor:
        return atomic_symbols_to_one_hot(
            [atom.GetSymbol() for atom in ligand.GetAtoms()]
        )

    @staticmethod
    def get_ligand_positions(ligand: Chem.Mol) -> Tensor:
        return torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)

    def __call__(self, complex: ProteinLigandComplex) -> Data:
        # remove all hydrogens from ligand
        ligand = complex.ligand.rdkit_mol()
        ligand = Chem.RemoveHs(ligand)

        x = self.get_ligand_atom_types(ligand)
        pos = self.get_ligand_positions(ligand)

        ligand_mask = torch.ones(len(x), dtype=torch.bool)
        scaffold_mask = get_ligand_scaffold_mask(ligand)
        if ligand_mask.sum() == 0:
            raise ValueError("No scaffold atom identified")

        return Data(
            x=x,
            pos=pos,
            ligand_mask=ligand_mask,
            scaffold_mask=scaffold_mask,
            ligand=ligand,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"
