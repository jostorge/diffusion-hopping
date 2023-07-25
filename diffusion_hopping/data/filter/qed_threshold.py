from rdkit.Chem import Descriptors

from diffusion_hopping.data.protein_ligand import ProteinLigandComplex


class QEDThresholdFilter:
    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def __call__(self, complex: ProteinLigandComplex) -> bool:
        mol = complex.ligand.rdkit_mol()
        return Descriptors.qed(mol) > self.threshold

    def __repr__(self) -> str:
        return f"QEDThresholdFilter(threshold={self.threshold})"

    def __str__(self) -> str:
        return self.__repr__()
