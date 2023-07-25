import unittest
from pathlib import Path

from diffusion_hopping.data import Protein, ProteinLigandComplex


class TestIsConnected(unittest.TestCase):
    def test_is_connected(self):
        from rdkit import Chem

        from diffusion_hopping.data.filter import IsConnectedFilter

        protein = Protein(Path("tests_data/complexes/1a0q/protein.pdb"))
        complex = ProteinLigandComplex(protein=protein, ligand=None)

        is_connected = IsConnectedFilter(cutoff=5)
        self.assertTrue(is_connected(complex))

    def test_is_not_connected(self):
        from diffusion_hopping.data.filter import IsConnectedFilter

        protein = Protein(Path("tests_data/complexes/1a0q/protein.pdb"))
        complex = ProteinLigandComplex(protein=protein, ligand=None)

        is_connected = IsConnectedFilter(cutoff=0)
        self.assertFalse(is_connected(complex))

    def test_representation(self):
        from diffusion_hopping.data.filter import IsConnectedFilter

        is_connected = IsConnectedFilter(cutoff=0)
        self.assertEqual(str(is_connected), "IsConnectedFilter(cutoff=0)")
