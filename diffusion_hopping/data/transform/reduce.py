import subprocess
import tempfile
from pathlib import Path


class ReduceTransform:
    def __init__(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir())

    def _run_reduce(self, options, input_path, output_path, expected_returncode):
        command = f"reduce {options} {str(input_path)} > {str(output_path)}"
        result = subprocess.run(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        if result.returncode != expected_returncode:
            raise RuntimeError(
                f"Reduce returned return code {result.returncode} when running '{command}'"
            )

    def __call__(self, path: Path) -> Path:
        temp_location = self.tmpdir / f"{path.stem}_reduce_tmp{path.suffix}"
        output_location = self.tmpdir / f"{path.stem}_reduce{path.suffix}"
        # it is weird, but apparently trim returns 255 if it works
        self._run_reduce("-Trim", path, temp_location, expected_returncode=255)
        self._run_reduce("-HIS", temp_location, output_location, expected_returncode=0)
        return output_location
