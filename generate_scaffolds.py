import argparse
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from diffusion_hopping.analysis.build import MoleculeBuilder
from diffusion_hopping.data import Protein, Ligand, ProteinLigandComplex
from diffusion_hopping.data.featurization import ProteinLigandSimpleFeaturization
from diffusion_hopping.data.transform import ObabelTransform, ReduceTransform
from diffusion_hopping.model import DiffusionHoppingModel


def parse_args():
    args = argparse.ArgumentParser(
        prog="generate_scaffolds.py",
        description="Generate scaffolds from input molecule and protein",
        epilog="Example: python generate_scaffolds.py --input_molecule input_molecule.sdf --input_protein input_protein.pdb --output output_folder",
    )
    args.add_argument(
        "--input_molecule", type=str, help="Input molecule", required=True
    )

    args.add_argument("--input_protein", type=str, help="Input protein", required=True)

    # output folder path, defaults to cwd
    args.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output folder path",
    )

    # optional argument for number of samples
    args.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples",
    )
    return args.parse_args()


def resolve_args(args):
    input_mol_path = Path(args.input_molecule)
    if not input_mol_path.exists():
        raise ValueError(f"{input_mol_path} does not exist")

    if (
        input_mol_path.suffix != ".sdf"
        and input_mol_path.suffix != ".mol2"
        and input_mol_path.suffix != ".pdb"
    ):
        raise ValueError(f"{input_mol_path} must be an sdf or mol2 file")
    input_protein_path = Path(args.input_protein)
    if not input_protein_path.exists():
        raise ValueError(f"{input_protein_path} does not exist")
    if input_protein_path.suffix != ".pdb":
        raise ValueError(f"{input_protein_path} must be a pdb file")

    output_folder_path = Path(args.output)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    num_samples = args.num_samples
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    return input_mol_path, input_protein_path, output_folder_path, num_samples


def main():
    args = parse_args()
    input_mol_path, input_protein_path, output_folder_path, num_samples = resolve_args(
        args
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path("checkpoints") / "gvp_conditional.ckpt"
    model = DiffusionHoppingModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    ).to(device)

    model.eval()
    model.freeze()

    ligand_transform_sdf = ObabelTransform(
        from_format=input_mol_path.suffix[1:], to_format="sdf"
    )
    protein_transform = Compose([ObabelTransform(), ReduceTransform()])

    protein, ligand = Protein(protein_transform(input_protein_path)), Ligand(
        ligand_transform_sdf(input_mol_path)
    )
    pl_complex = ProteinLigandComplex(protein, ligand, identifier="complex")
    featurization = ProteinLigandSimpleFeaturization(
        c_alpha_only=True, cutoff=8.0, mode="residue"
    )

    batch = Batch.from_data_list([featurization(pl_complex)] * num_samples).to(
        model.device
    )

    sample_results = model.model.sample(batch)
    final_output = sample_results[-1]

    molecule_builder = MoleculeBuilder(include_invalid=False)
    molecules: list[Chem.Mol] = molecule_builder(final_output)

    # save mol to sdf file
    for i, mol in enumerate(molecules):
        path = output_folder_path / f"output_{i}.sdf"
        Chem.MolToMolFile(mol, str(path))


if __name__ == "__main__":
    main()
