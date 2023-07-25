import unittest
from pathlib import Path

from diffusion_hopping.data.transform import ReduceTransform
from diffusion_hopping.util import disable_obabel_and_rdkit_logging


class TestReduce(unittest.TestCase):
    def test_reduce(self):
        disable_obabel_and_rdkit_logging()
        transform = ReduceTransform()
        path = Path("tests_data/complexes/1a0q/protein.pdb")
        output_location = transform(path)
        self.assertFalse(path == output_location)
        self.assertTrue(output_location.exists())

        output_location.unlink()
