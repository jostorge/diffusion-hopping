import pickle
import re
import unicodedata
from pathlib import Path
from typing import Iterator, List

from diffusion_hopping.data.protein_ligand import ProteinLigandComplex


class LMDBStorage:
    def __init__(self, path: Path, readonly=True) -> None:
        self.path = path
        self.db = None
        self._readonly = readonly

    def __contains__(self, key):
        self._connect()
        with self.db.begin() as txn:
            return txn.get(key.encode()) is not None

    def _connect(self):
        import lmdb

        if self.db is None:
            self.db = lmdb.open(
                str(self.path),
                map_size=1024**4,
                subdir=False,
                readonly=self._readonly,
            )

    def __len__(self):
        self._connect()
        with self.db.begin() as txn:
            return txn.stat()["entries"]

    def __getitem__(self, key: str):
        self._connect()
        with self.db.begin() as txn:
            return pickle.loads(txn.get(key.encode()))

    def __setitem__(self, key: str, value):
        self._connect()
        with self.db.begin(write=True) as txn:
            txn.put(key.encode(), pickle.dumps(value))

    def __iter__(self):
        self._connect()
        with self.db.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None


class ProcessedComplexStorage:
    def __init__(self, path: Path):
        self.path = path

    def __len__(self):
        return len(list(self.path.iterdir()))

    def __contains__(self, index):
        return (
            (self.path / index).exists()
            and (self.path / index).is_dir()
            and (self.path / index / f"protein.pdb").exists()
            and (self.path / index / f"ligand.sdf").exists()
        )

    def __getitem__(self, index):
        return ProteinLigandComplex.from_file(self.path / index, index)

    def __setitem__(self, index, value):
        value.to_file(self.path / index)

    def __iter__(self) -> Iterator[str]:
        for complex in self.path.iterdir():
            if complex.name in self:
                yield complex.name


def keys_from_file(file: Path) -> List[str]:
    return [item.strip() for item in file.read_text().split()]


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/a52bdea5a27ba44b13eda93642231c65c581e083/django/utils/text.py#LL420-L437C53
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")
