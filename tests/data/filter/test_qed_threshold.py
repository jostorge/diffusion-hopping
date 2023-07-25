import unittest
from types import SimpleNamespace

from rdkit import Chem
from rdkit.Chem import Descriptors

from diffusion_hopping.data import ProteinLigandComplex
from diffusion_hopping.data.filter import QEDThresholdFilter


class TestQEDThresholdFilter(unittest.TestCase):
    def test_returns_false_on_worse_molecule(self):
        mol = Chem.MolFromSmiles("CC(=O)N")
        threshold = 0.5
        qed = Descriptors.qed(mol)
        ligand = SimpleNamespace(rdkit_mol=lambda: mol)
        complex = ProteinLigandComplex(ligand=ligand, protein=None)
        filter = QEDThresholdFilter(threshold=threshold)
        self.assertLess(qed, threshold)
        self.assertFalse(filter(complex))

    def test_returns_true_on_better_molecule(self):
        mol = Chem.MolFromSmiles("CC(=O)N")
        threshold = 0.1
        qed = Descriptors.qed(mol)
        ligand = SimpleNamespace(rdkit_mol=lambda: mol)
        complex = ProteinLigandComplex(ligand=ligand, protein=None)
        filter = QEDThresholdFilter(threshold=threshold)
        self.assertGreater(qed, threshold)
        self.assertTrue(filter(complex))

    def test_representation(self):
        threshold = 0.1
        filter = QEDThresholdFilter(threshold=threshold)
        self.assertEqual(str(filter), f"QEDThresholdFilter(threshold={threshold})")
