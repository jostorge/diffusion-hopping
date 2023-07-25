import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from biopandas.pdb.engines import amino3to1dict
from rdkit import Chem
from torch_geometric.data import HeteroData

from diffusion_hopping.data.featurization.util import (
    atom_names,
    atomic_symbols_to_one_hot,
    get_ligand_scaffold_mask,
    residue_names,
    residue_names_to_one_hot,
)
from diffusion_hopping.data.protein_ligand import ProteinLigandComplex
from diffusion_hopping.data.transform import ChainSelectionTransform


class ProteinLigandSimpleFeaturization:
    def __init__(
        self, cutoff=10, c_alpha_only=False, center_complex=False, mode="chain"
    ) -> None:
        self.c_alpha_only = c_alpha_only
        self.center_complex = center_complex
        self.cutoff = cutoff
        self.mode = mode
        self.chain_selection_transform = ChainSelectionTransform(
            cutoff=cutoff, mode=mode
        )
        self.protein_features = len(residue_names) if c_alpha_only else len(atom_names)
        self.ligand_features = len(atom_names)

    @staticmethod
    def get_ligand_atom_types(ligand: Chem.Mol) -> torch.Tensor:
        return atomic_symbols_to_one_hot(
            [atom.GetSymbol() for atom in ligand.GetAtoms()]
        )

    @staticmethod
    def get_protein_atom_types(protein: pd.DataFrame) -> torch.Tensor:
        return atomic_symbols_to_one_hot(protein["element_symbol"].values)

    def get_protein_residues(self, protein: pd.DataFrame) -> torch.Tensor:
        residue_names = protein["residue_name"].map(amino3to1dict).values
        return residue_names_to_one_hot(residue_names)

    @staticmethod
    def get_ligand_positions(ligand: Chem.Mol) -> torch.Tensor:
        return torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)

    @staticmethod
    def get_protein_positions(protein: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(
            protein[["x_coord", "y_coord", "z_coord"]].values, dtype=torch.float
        )

    def __call__(self, complex: ProteinLigandComplex) -> HeteroData:
        # remove all hydrogens from ligand
        ligand = complex.ligand.rdkit_mol()
        ligand = Chem.RemoveHs(ligand)

        x_ligand = self.get_ligand_atom_types(ligand)
        pos_ligand = self.get_ligand_positions(ligand)

        protein: PandasPdb = complex.protein.pandas_pdb()
        protein: PandasPdb = self.chain_selection_transform(protein, pos_ligand)

        if self.c_alpha_only:
            protein_df = protein.get("c-alpha")
            protein_df = protein_df[protein_df["residue_name"].isin(amino3to1dict)]
            x_protein = self.get_protein_residues(protein_df)
        else:
            # remove all hydrogens from protein
            protein_df = protein.get("hydrogen", invert=True)
            x_protein = self.get_protein_atom_types(protein_df)

        pos_protein = self.get_protein_positions(protein_df)

        scaffold_mask = get_ligand_scaffold_mask(ligand)
        # if scaffold_mask.sum() == 0:
        #    raise ValueError("No scaffold atom identified")
        # elif scaffold_mask.sum() == torch.numel(scaffold_mask):
        #    raise ValueError("Only scaffold atoms identified")

        if self.center_complex:
            dtype = pos_ligand.dtype
            mean_pos = pos_ligand.to(torch.float64).mean(dim=0)
            pos_ligand = (pos_ligand.to(torch.float64) - mean_pos).to(dtype)
            pos_protein = (pos_protein.to(torch.float64) - mean_pos).to(dtype)

        return HeteroData(
            ligand={
                "x": x_ligand,
                "pos": pos_ligand,
                "scaffold_mask": scaffold_mask,
                "ref": ligand,
                "path": complex.ligand.path,
            },
            protein={
                "x": x_protein,
                "pos": pos_protein,
                # "ref": protein_df, # maybe add protein df
                "path": complex.protein.path,
            },
            identifier=complex.identifier,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, c_alpha_only={self.c_alpha_only}, center_complex={self.center_complex}, mode={self.mode})"
