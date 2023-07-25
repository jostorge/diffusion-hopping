import unittest

from diffusion_hopping.util import disable_obabel_and_rdkit_logging


class TestGeneralUtils(unittest.TestCase):
    def test_disable_obabel_and_rdkit_logging(self):
        from openbabel import openbabel

        disable_obabel_and_rdkit_logging()
        self.assertEqual(openbabel.obErrorLog.GetOutputLevel(), 0)
