import subprocess
from pathlib import Path

from diffusion_hopping.analysis.evaluate.util import _run_commands


def qvina_score(row, size=20.0, exhaustiveness=16):
    try:
        if row["molecule"] is None:
            return None
        protein_pdbqt = _prepare_protein(row["test_set_item"]["protein"].path)
        ligand_pdbqt = _prepare_ligand(row["molecule_path"])
        return _calculate_qvina_score(
            protein_pdbqt,
            ligand_pdbqt,
            row["molecule"],
            size=size,
            exhaustiveness=exhaustiveness,
        )
    except:
        return None


def _prepare_protein(protein_path) -> Path:
    protein_folder = protein_path.parent.resolve()
    protein_name = protein_path.name
    protein_pdbqt = f"{protein_path.stem}.pdbqt"
    protein_pdbqt = protein_folder / protein_pdbqt

    commands = [
        'eval "$(conda shell.zsh hook)"',
        "conda activate mgltools",
        f"cd {protein_folder}",
        f"prepare_receptor4.py -r {protein_name} -o {protein_pdbqt.name}",
    ]
    _run_commands(commands)
    return protein_pdbqt


def _prepare_ligand(ligand_path) -> Path:
    ligand_folder = ligand_path.parent.resolve()
    ligand_name = ligand_path.name
    ligand_pdbqt = f"{ligand_path.stem}.pdbqt"
    ligand_pdbqt = ligand_folder / ligand_pdbqt

    commands = [
        'eval "$(conda shell.zsh hook)"',
        "conda activate mgltools",
        f"cd {ligand_folder}",
        f"prepare_ligand4.py -l {ligand_name} -o {ligand_pdbqt.name}",
    ]
    _run_commands(commands)
    return ligand_pdbqt


def _calculate_qvina_score(
    protein_path, ligand_path, mol, size=20.0, exhaustiveness=16
) -> float:
    center = mol.GetConformer().GetPositions().mean(axis=0)
    command = f"qvina2.1 --receptor {protein_path.resolve()} --ligand {ligand_path.resolve()} --center_x {center[0]} --center_y {center[1]} --center_z {center[2]} --size_x {size} --size_y {size} --size_z {size} --exhaustiveness {exhaustiveness} "
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"QVina returned non-zero return code {result.returncode} when running '{command}'"
        )
    out_lines = iter(result.stdout.splitlines())
    for line in out_lines:
        if line.startswith("-----+------------+----------+----------"):
            break

    relevant_line = next(out_lines)
    # print("Identified line", relevant_line)
    if relevant_line.split()[0] != "1":
        raise RuntimeError("No valid result found")
    score = float(relevant_line.split()[1])
    return score
