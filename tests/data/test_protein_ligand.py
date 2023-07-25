import os
import tempfile
import unittest
from pathlib import Path

from diffusion_hopping.data.protein_ligand import Ligand, Protein, ProteinLigandComplex


class TestProteinLigand(unittest.TestCase):
    def test_protein_stores_path(self):
        path = "tests_data/complexes/1a0q/protein.pdb"
        protein = Protein(path)
        self.assertEqual(protein.path, path)

    def test_creating_protein_without_path_fails(self):
        with self.assertRaises(TypeError):
            Protein()

    def test_creating_protein_with_invalid_path_fails(self):
        with self.assertRaises(FileNotFoundError):
            Protein("invalid/path")

    def test_protein_is_read_correctly(self):
        path = "tests_data/complexes/1a0q/protein.pdb"
        protein = Protein(path)
        df = protein.pandas_pdb()
        self.assertEqual(df.df["ATOM"]["residue_name"][0], "ILE")
        self.assertEqual(df.df["ATOM"]["residue_number"][0], 2)
        self.assertEqual(df.df["ATOM"]["atom_name"][0], "N")
        self.assertEqual(df.df["ATOM"]["element_symbol"][0], "N")
        self.assertEqual(df.df["ATOM"]["x_coord"][0], 27.234)
        self.assertEqual(len(df.df["ATOM"]), 6284)

    def test_ligand_stores_path(self):
        path = "tests_data/complexes/1a0q/ligand.sdf"
        ligand = Ligand(path)
        self.assertEqual(ligand.path, path)

    def test_creating_ligand_without_path_fails(self):
        with self.assertRaises(TypeError):
            Ligand()

    def test_creating_ligand_with_invalid_path_fails(self):
        with self.assertRaises(FileNotFoundError):
            Ligand("invalid/path")

    def test_ligand_is_read_correctly(self):
        path = "tests_data/complexes/1a0q/ligand.sdf"
        ligand = Ligand(path)
        mol = ligand.rdkit_mol()
        self.assertEqual(mol.GetNumAtoms(), 23)
        self.assertEqual(mol.GetNumBonds(), 23)
        self.assertEqual(mol.GetAtomWithIdx(0).GetSymbol(), "C")
        self.assertEqual(mol.GetAtomWithIdx(0).GetFormalCharge(), 0)

    def test_protein_ligand_complex_stores_properties(self):
        protein_path = "tests_data/complexes/1a0q/protein.pdb"
        ligand_path = "tests_data/complexes/1a0q/ligand.sdf"
        protein = Protein(protein_path)
        ligand = Ligand(ligand_path)

        complex = ProteinLigandComplex(protein, ligand)
        self.assertEqual(complex.protein, protein)
        self.assertEqual(complex.ligand, ligand)

    def test_creating_protein_ligand_complex_without_paths_fails(self):
        with self.assertRaises(TypeError):
            ProteinLigandComplex()

    def test_protein_ligand_complex_is_read_correctly(self):
        protein_path = "tests_data/complexes/1a0q/protein.pdb"
        ligand_path = "tests_data/complexes/1a0q/ligand.sdf"
        protein = Protein(protein_path)
        ligand = Ligand(ligand_path)

        complex = ProteinLigandComplex(protein, ligand)

        df = complex.protein.pandas_pdb().df["ATOM"]
        ref_df = Protein(protein_path).pandas_pdb().df["ATOM"]
        self.assertTrue(df.equals(ref_df))

    def test_protein_ligand_storing_works(self):
        protein_path = "tests_data/complexes/1a0q/protein.pdb"
        ligand_path = "tests_data/complexes/1a0q/ligand.sdf"
        protein = Protein(protein_path)
        ligand = Ligand(ligand_path)

        complex = ProteinLigandComplex(protein, ligand)

        with tempfile.TemporaryDirectory() as tmpdir:
            complex.to_file(Path(tmpdir))
            complex2 = ProteinLigandComplex.from_file(Path(tmpdir))

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "protein.pdb")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "ligand.sdf")))

            df1 = complex.protein.pandas_pdb().df["ATOM"]
            df2 = complex2.protein.pandas_pdb().df["ATOM"]
            self.assertTrue(df1.equals(df2))

            mol1 = complex.ligand.rdkit_mol()
            mol2 = complex2.ligand.rdkit_mol()
            self.assertEqual(mol1.GetNumAtoms(), mol2.GetNumAtoms())

    def test_protein_ligand_complex_has_identifier(self):
        protein_path = "tests_data/complexes/1a0q/protein.pdb"
        ligand_path = "tests_data/complexes/1a0q/ligand.sdf"
        protein = Protein(protein_path)
        ligand = Ligand(ligand_path)

        complex = ProteinLigandComplex(protein, ligand, identifier="1a0q")
        self.assertEqual(complex.identifier, "1a0q")
