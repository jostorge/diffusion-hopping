import unittest

import torch

from diffusion_hopping.data import Ligand
from diffusion_hopping.data.featurization.util import (
    atomic_symbols_to_one_hot,
    get_ligand_scaffold_mask,
    one_hot,
    residue_names_to_one_hot,
)


class TestUtil(unittest.TestCase):
    def test_get_ligand_scaffold_mask(self):
        ligand = Ligand("tests_data/complexes/1a0q/ligand.sdf")
        mol = ligand.rdkit_mol()
        mask = get_ligand_scaffold_mask(mol)

        self.assertEqual(mask.shape, (mol.GetNumAtoms(),))
        self.assertEqual(mask.sum(), 6)
        self.assertEqual(mask.dtype, torch.bool)

    def test_atomic_symbols_to_one_hot(self):
        symbols = ["N", "C", "O", "S", "B", "Br", "Cl", "P", "I", "F"]
        one_hot = atomic_symbols_to_one_hot(symbols)

        self.assertEqual(one_hot.shape, (10, 10))
        self.assertEqual(one_hot.dtype, torch.int64)
        self.assertEqual(one_hot.sum(dim=1).tolist(), [1] * 10)
        self.assertEqual(one_hot.sum(dim=0).tolist(), [1] * 10)
        self.assertEqual(one_hot[0][0], 0)
        self.assertEqual(one_hot[0][1], 1)
        self.assertEqual(one_hot[1][0], 1)
        self.assertEqual(one_hot[1][1], 0)

    def test_residue_names_to_one_hot(self):
        symbols = [
            "C",
            "A",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        one_hot = residue_names_to_one_hot(symbols)

        self.assertEqual(one_hot.shape, (20, 20))
        self.assertEqual(one_hot.dtype, torch.int64)
        self.assertEqual(one_hot.sum(dim=1).tolist(), [1] * 20)
        self.assertEqual(one_hot.sum(dim=0).tolist(), [1] * 20)
        self.assertEqual(one_hot[0][0], 0)
        self.assertEqual(one_hot[0][1], 1)
        self.assertEqual(one_hot[1][0], 1)
        self.assertEqual(one_hot[1][1], 0)

    def test_one_hot(self):
        symbols = ["A", "B", "C", "D", "E"]
        query = ["A", "A", "C", "E", "E", "A"]
        one_hot_output = one_hot(query, symbols)

        self.assertEqual(one_hot_output.shape, (6, 5))
        self.assertEqual(one_hot_output.dtype, torch.int64)
        self.assertEqual(one_hot_output.sum(dim=1).tolist(), [1] * 6)
        self.assertEqual(one_hot_output.sum(dim=0).tolist(), [3, 0, 1, 0, 2])
