import tempfile
import unittest
from pathlib import Path

from diffusion_hopping.data import ProteinLigandComplex
from diffusion_hopping.data.util import ProcessedComplexStorage, keys_from_file, slugify


class TestProcessedComplexStorage(unittest.TestCase):
    def test_keys_from_file(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write("key1".encode())
            f.flush()
            keys = keys_from_file(Path(f.name))
            self.assertEqual(keys, ["key1"])

    def test_slugify(self):
        self.assertEqual(slugify("1a0q"), "1a0q")
        self.assertEqual(slugify("1a0q.pdb"), "1a0qpdb")
        self.assertEqual(slugify("C:/abc/1a0q.pdb.gz"), "cabc1a0qpdbgz")

    def test_storage_contains(self):
        complex_path = Path("tests_data/complexes/1a0q")
        pl_complex = ProteinLigandComplex.from_file(complex_path, "1a0q")

        with tempfile.TemporaryDirectory() as d:
            storage = ProcessedComplexStorage(Path(d))
            self.assertFalse("1a0q" in storage)
            storage["1a0q"] = pl_complex
            self.assertTrue("1a0q" in storage)

    def test_storage_put_and_get(self):
        complex_path = Path("tests_data/complexes/1a0q")
        pl_complex = ProteinLigandComplex.from_file(complex_path, "1a0q")

        with tempfile.TemporaryDirectory() as d:
            storage = ProcessedComplexStorage(Path(d))
            storage["1a0q"] = pl_complex

            stored_complex = storage["1a0q"]
            protein_text = stored_complex.protein.path.read_text()
            ligand_text = stored_complex.ligand.path.read_text()
            self.assertEqual(protein_text, pl_complex.protein.path.read_text())
            self.assertEqual(ligand_text, pl_complex.ligand.path.read_text())

    def test_storage_from_existing_folder(self):
        storage = ProcessedComplexStorage(Path("tests_data/complexes/"))
        self.assertTrue("1a0q" in storage)
        self.assertFalse("1a0r" in storage)
        self.assertEqual(len(storage), 1)
