import base64
import subprocess
from io import BytesIO
from typing import List

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw


def _to_smiles(row):
    try:
        if row["molecule"] is None:
            return None
        return Chem.MolToSmiles(row["molecule"])
    except:
        return None


def _image_with_highlighted_atoms(mol, atoms_to_highlight):
    try:
        if mol is None:
            return None
        bonds_to_highlight = [
            bond.GetIdx()
            for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in atoms_to_highlight
            or bond.GetEndAtomIdx() in atoms_to_highlight
        ]
        return Draw.MolToImage(
            mol,
            size=(200, 200),
            highlightAtoms=atoms_to_highlight,
            highlightBonds=bonds_to_highlight,
        )
    except:
        return None


def _to_smiles_image(row):
    if row["SMILES"] is None:
        return None
    mol = Chem.MolFromSmiles(row["SMILES"])
    try:
        # We need this as there is a bug in RDKit that causes the program to crash on some molecules
        return Draw.MolToImage(mol, size=(200, 200))
    except:
        return None


def _run_commands(commands: List[str]) -> str:
    commands = "\n".join(commands)
    # execute command in shell
    proc = subprocess.Popen(
        "/bin/bash",
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdin.write(commands.encode("utf-8"))
    proc.stdin.close()
    result = proc.wait()

    if result != 0:
        raise RuntimeError(
            f"_run_commands return code {result} when running '{commands}'"
        )
    return "\n".join([line.decode("utf-8") for line in proc.stdout.readlines()])


def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, "jpeg")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    try:
        if not isinstance(im, Image.Image):
            raise ValueError
        return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
    except:
        return ""


def to_html(df, path=None, image_columns=None):
    if image_columns is None:
        image_columns = []
    return df.to_html(
        buf=path,
        formatters={key: image_formatter for key in image_columns},
        escape=False,
    )
