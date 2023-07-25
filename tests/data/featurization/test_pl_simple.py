import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from diffusion_hopping.data import Ligand, Protein, ProteinLigandComplex
from diffusion_hopping.data.featurization.pl_simple import (
    ProteinLigandSimpleFeaturization,
)
from diffusion_hopping.data.featurization.util import atomic_symbols_to_one_hot


class TestProteinLigandSimpleFeaturization(unittest.TestCase):
    def test_representation1(self):
        featurization = ProteinLigandSimpleFeaturization()
        self.assertEqual(
            featurization.__repr__(),
            "ProteinLigandSimpleFeaturization(cutoff=10, c_alpha_only=False, center_complex=False, mode=chain)",
        )

    def test_representation2(self):
        featurization = ProteinLigandSimpleFeaturization(
            cutoff=20, c_alpha_only=True, mode="residue"
        )
        self.assertEqual(
            featurization.__repr__(),
            "ProteinLigandSimpleFeaturization(cutoff=20, c_alpha_only=True, center_complex=False, mode=residue)",
        )

    def test_get_ligand_atom_types(self):
        featurization = ProteinLigandSimpleFeaturization()
        ligand_path = "tests_data/complexes/1a0q/ligand.sdf"
        mol = Ligand(ligand_path).rdkit_mol()
        atom_types = featurization.get_ligand_atom_types(mol)
        atom_types_ref = atomic_symbols_to_one_hot(
            [atom.GetSymbol() for atom in mol.GetAtoms()]
        )
        self.assertEqual(atom_types.shape, (mol.GetNumAtoms(), 10))
        self.assertEqual(atom_types.dtype, torch.int64)
        self.assertTrue(torch.all(atom_types == atom_types_ref))

    def test_get_protein_atom_types(self):
        featurization = ProteinLigandSimpleFeaturization()
        df = pd.DataFrame({"element_symbol": ["C", "N", "O", "S"]})
        atom_types = featurization.get_protein_atom_types(df)
        atom_types_ref = atomic_symbols_to_one_hot(df["element_symbol"].tolist())
        self.assertEqual(atom_types.dtype, torch.int64)
        self.assertTrue(torch.all(atom_types == atom_types_ref))

    def test_get_ligand_positions(self):
        featurization = ProteinLigandSimpleFeaturization()
        ligand_path = "tests_data/complexes/1a0q/ligand.sdf"
        mol = Ligand(ligand_path).rdkit_mol()
        pos = featurization.get_ligand_positions(mol)
        self.assertEqual(pos.shape, (mol.GetNumAtoms(), 3))
        self.assertEqual(pos.dtype, torch.float32)
        self.assertEqual(pos[0, 0], 13.3050)

    def test_get_protein_positions(self):
        featurization = ProteinLigandSimpleFeaturization()
        protein_path = "tests_data/complexes/1a0q/protein.pdb"
        df = Protein(protein_path).pandas_pdb().df["ATOM"]
        pos = featurization.get_protein_positions(df)
        self.assertEqual(pos.shape, (df.shape[0], 3))
        self.assertEqual(pos.dtype, torch.float32)
        self.assertEqual(pos[0, 0], 27.2340)

    def test_featurization(self):
        featurization = ProteinLigandSimpleFeaturization(
            cutoff=20, c_alpha_only=True, mode="residue"
        )
        complex_path = Path("tests_data/complexes/1a0q")
        complex = ProteinLigandComplex.from_file(complex_path, "1a0q")
        data = featurization(complex)
        self.assertIsInstance(data, HeteroData)
        self.assertEqual(data.num_nodes, 209)
        self.assertEqual(data.identifier, "1a0q")
        self.assertEqual(data["ligand"].path, complex_path / "ligand.sdf")
        self.assertEqual(data["protein"].path, complex_path / "protein.pdb")
        self.assertEqual(data["ligand"].num_nodes, 23)
        self.assertEqual(data["protein"].num_nodes, 186)

        self.assertEqual(data["ligand"].x.shape, (23, 10))
        self.assertEqual(data["ligand"].x.dtype, torch.int64)
        self.assertEqual(data["ligand"].pos.shape, (23, 3))
        self.assertEqual(data["ligand"].pos.dtype, torch.float32)

        self.assertEqual(data["protein"].x.shape, (186, 20))
        self.assertEqual(data["protein"].x.dtype, torch.int64)
        self.assertEqual(data["protein"].pos.shape, (186, 3))
        self.assertEqual(data["protein"].pos.dtype, torch.float32)

    def test_featurization_centers(self):
        featurization = ProteinLigandSimpleFeaturization(
            cutoff=20, c_alpha_only=False, mode="residue", center_complex=True
        )
        complex_path = Path("tests_data/complexes/1a0q")
        complex = ProteinLigandComplex.from_file(complex_path, "1a0q")
        data = featurization(complex)

        ligand_pos = data["ligand"].pos
        self.assertAlmostEqual(float(ligand_pos[:, 0].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(ligand_pos[:, 1].mean()), 0.0, places=4)
        self.assertAlmostEqual(float(ligand_pos[:, 2].mean()), 0.0, places=4)
