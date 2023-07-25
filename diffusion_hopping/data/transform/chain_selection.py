import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy import spatial


class ChainSelectionTransform:
    def __init__(self, cutoff: float = 10.0, mode: str = "chain") -> None:
        self.tmpdir = Path(tempfile.gettempdir())
        self.cutoff = cutoff
        assert mode in ["chain", "residue"]
        self.mode = mode

    def _prepare_output(self, protein, df):
        df["atom_number"] = np.arange(1, len(df) + 1)

        protein.df["ATOM"] = df[df.record_name == "ATOM"]
        protein.df["HETATM"] = df[df.record_name == "HETATM"]
        if "OTHERS" in protein.df:
            protein.df["OTHERS"].drop(protein.df["OTHERS"].index, inplace=True)

        """chains = df["chain_id"].drop_duplicates()
        selection = " or ".join(
            map(lambda c: f"chain {c if c.strip() != '' else '_'}", chains)
        )
        ellipsis = lambda s: s[:15] + "..." + s[-15:] if len(s) > 33 else s
        protein.df["OTHERS"] = pd.DataFrame(
            {
                "record_name": ["REMARK"],
                "entry": [f" Selection '{ellipsis(selection)}'"],
                "line_idx": [-1],
            }
        )"""

        return protein

    def __call__(self, protein: PandasPdb, ligand_coords: np.array) -> PandasPdb:

        df = pd.concat([protein.df["ATOM"], protein.df["HETATM"]], ignore_index=True)
        df.sort_values(by="line_idx", inplace=True)
        # Remove all chains where there is a residue named HOH, to remove water chains
        is_water_chain = df.groupby("chain_id")["residue_name"].transform(
            frozenset({"HOH"}).issubset
        )
        df = df[~is_water_chain].copy()

        # Only consider proper residues (amino acids) for the distance filtering
        is_residue = df.groupby(["chain_id", "residue_number"])["atom_name"].transform(
            frozenset({"CA", "N", "C"}).issubset
        )

        filter_by_cutoff = (
            lambda x: spatial.distance.cdist(
                ligand_coords,
                x.loc[is_residue, ["x_coord", "y_coord", "z_coord"]].values,
            ).min(initial=np.inf)
            < self.cutoff
        )
        if self.mode == "chain":
            df = df.groupby("chain_id", sort=False).filter(filter_by_cutoff)
        elif self.mode == "residue":
            df = df.groupby(["chain_id", "residue_number"], sort=False).filter(
                filter_by_cutoff
            )

        protein = self._prepare_output(protein, df)
        return protein
