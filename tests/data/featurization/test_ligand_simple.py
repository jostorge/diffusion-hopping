import unittest
from types import SimpleNamespace

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from diffusion_hopping.data.featurization.ligand_simple import LigandSimpleFeaturization
from diffusion_hopping.data.featurization.util import (
    atomic_symbols_to_one_hot,
    get_ligand_scaffold_mask,
)
from diffusion_hopping.data.protein_ligand import ProteinLigandComplex


class TestLigandSimpleFeaturization(unittest.TestCase):
    def test_get_ligand_atom_types(self):

        ligand = Chem.MolFromSmiles("CC(=O)N")
        ligand = Chem.RemoveHs(ligand)
        x = LigandSimpleFeaturization().get_ligand_atom_types(ligand)
        self.assertEqual(x.shape, (4, 10))
        self.assertEqual(x.dtype, torch.int64)

    def test_get_ligand_positions(self):
        ligand = Chem.MolFromSmiles("CC(=O)N")
        ligand = Chem.AddHs(ligand)
        AllChem.EmbedMolecule(ligand)
        x = LigandSimpleFeaturization().get_ligand_positions(ligand)
        self.assertEqual(x.shape, (9, 3))
        self.assertEqual(x.dtype, torch.float32)

    def test_call(self):
        ligand = Chem.MolFromSmiles("CC(=O)N")
        ligand = Chem.AddHs(ligand)
        AllChem.EmbedMolecule(ligand)
        ligand_struct = SimpleNamespace(rdkit_mol=lambda: ligand)
        complex = ProteinLigandComplex(ligand=ligand_struct, protein=None)
        data = LigandSimpleFeaturization()(complex)
        self.assertEqual(data.x.shape, (4, 10))
        self.assertEqual(data.x.dtype, torch.int64)
        self.assertEqual(data.pos.shape, (4, 3))
        self.assertEqual(data.pos.dtype, torch.float32)
        self.assertEqual(data.ligand_mask.shape, (4,))
        self.assertEqual(data.ligand_mask.dtype, torch.bool)
        self.assertEqual(data.scaffold_mask.shape, (4,))
        self.assertEqual(data.scaffold_mask.dtype, torch.bool)
