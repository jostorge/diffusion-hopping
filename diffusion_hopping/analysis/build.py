from pathlib import Path
from tempfile import NamedTemporaryFile

from rdkit import Chem
from torch_geometric.data import HeteroData

from diffusion_hopping.data import Ligand
from diffusion_hopping.data.featurization.util import atom_names
from diffusion_hopping.data.transform import ObabelTransform


class MoleculeBuilder:
    def __init__(
        self,
        include_invalid=False,
        sanitize=True,
        removeHs=True,
        fix_hydrogens=True,
        atom_names=atom_names,
    ):
        self.include_invalid = include_invalid
        self.sanitize = sanitize
        self.removeHs = removeHs
        self.should_fix_hydrogens = fix_hydrogens
        self.xyz_to_sdf = ObabelTransform(from_format="xyz", to_format="sdf")
        self.atom_names = atom_names

    def __call__(self, x):
        molecules = []
        for item in x.to_data_list():
            try:
                mol = self.build_mol(item["ligand"])
                molecules.append(mol)
            except ValueError as e:
                if self.include_invalid:
                    molecules.append(None)
        return molecules

    def build_mol(self, ligand: HeteroData) -> Chem.Mol:
        """Build molecules from HeteroData"""
        xyz_path = self.xyz_from_hetero_data(ligand)
        sdf_path = self.xyz_to_sdf(xyz_path)
        xyz_path.unlink()
        mol = self.mol_from_sdf(sdf_path)
        sdf_path.unlink()
        if self.should_fix_hydrogens:
            self.fix_hydrogens(mol)

        return mol

    def xyz_from_hetero_data(self, x: HeteroData) -> Path:
        """Build xyz from HeteroData"""
        pos = x.pos.detach().cpu().numpy()
        types = x.x.detach().cpu().argmax(axis=-1).numpy()
        types = [self.atom_names[t] for t in types]
        return self.write_xyz_file(pos, types)

    def mol_from_sdf(self, sdf_path: Path) -> Chem.Mol:
        """Build molecule from sdf file"""
        return Ligand(sdf_path).rdkit_mol(
            sanitize=self.sanitize, removeHs=self.removeHs
        )

    @staticmethod
    def write_xyz_file(pos, atom_type) -> Path:
        with NamedTemporaryFile("w", delete=False) as f:
            f.write(f"{len(pos)}\n")
            f.write("generated by model\n")
            for pos, atom_type in zip(pos, atom_type):
                f.write(f"{atom_type} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:,.9f}\n")
            return Path(f.name)

    @staticmethod
    def fix_hydrogens(mol: Chem.Mol):
        organicSubset = (5, 6, 7, 8, 9, 15, 16, 17, 35, 53)
        for at in mol.GetAtoms():
            if at.GetAtomicNum() not in organicSubset:
                continue
            at.SetNoImplicit(False)
            at.SetNumExplicitHs(0)
            at.SetNumRadicalElectrons(0)
        Chem.SanitizeMol(mol)
        return mol