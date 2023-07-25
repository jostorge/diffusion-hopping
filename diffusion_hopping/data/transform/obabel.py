import tempfile
from pathlib import Path

from openbabel import openbabel


class ObabelTransform:
    def __init__(self, from_format="pdb", to_format="pdb") -> None:
        self.tmpdir = Path(tempfile.gettempdir())
        self.obConversion = openbabel.OBConversion()
        self.to_format = to_format
        self.obConversion.SetInAndOutFormats(from_format, to_format)

    def __call__(self, path: Path) -> Path:
        mol = openbabel.OBMol()
        self.obConversion.ReadFile(mol, str(path))
        output_location = self.tmpdir / f"{path.stem}_obabel.{self.to_format}"
        self.obConversion.WriteFile(mol, str(output_location))

        return output_location
