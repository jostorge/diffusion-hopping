import unittest
from pathlib import Path

from diffusion_hopping.data.transform import ObabelTransform
from diffusion_hopping.util import disable_obabel_and_rdkit_logging


class TestObabel(unittest.TestCase):
    def test_obabel_pdb(self):

        disable_obabel_and_rdkit_logging()
        transform = ObabelTransform(from_format="pdb", to_format="pdb")
        path = Path("tests_data/complexes/1a0q/protein.pdb")
        output_location = transform(path)
        self.assertFalse(path == output_location)
        self.assertTrue(output_location.exists())
        output_location.unlink()

    def test_obabel_conversion(self):
        disable_obabel_and_rdkit_logging()
        transform = ObabelTransform(from_format="pdb", to_format="sdf")
        path = Path("tests_data/complexes/1a0q/protein.pdb")
        output_location = transform(path)
        self.assertFalse(path == output_location)
        self.assertTrue(output_location.exists())
        self.assertTrue(output_location.suffix == ".sdf")
        output_location.unlink()
