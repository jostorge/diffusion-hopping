import warnings

from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule


class LargestFragmentTransform:
    def __init__(self):
        pass

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        return max(
            Chem.GetMolFrags(mol, asMols=True),
            key=lambda x: x.GetNumAtoms(),
            default=mol,
        )


class UniversalForceFieldTransform:
    def __init__(self, max_iters=200, sanitize=True):
        self.max_iters = max_iters
        self.sanitize = sanitize

    def __call__(self, mol: Chem.Mol) -> float:
        mol = Chem.Mol(mol)
        did_converge = UFFOptimizeMolecule(mol, maxIters=self.max_iters)
        if did_converge != 0:
            warnings.warn(
                f"Maximum number of Universal Force Field iterations reached."
                f"Returning molecule after {self.max_iters} steps."
            )
        if self.sanitize:
            Chem.SanitizeMol(mol)
        return mol
