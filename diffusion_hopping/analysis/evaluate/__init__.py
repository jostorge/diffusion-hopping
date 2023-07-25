import functools
import itertools
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from diffusion_hopping.analysis.build import MoleculeBuilder
from diffusion_hopping.analysis.evaluate.qvina import qvina_score
from diffusion_hopping.analysis.evaluate.util import (
    _image_with_highlighted_atoms,
    _to_smiles,
    _to_smiles_image,
    image_formatter,
    to_html,
)
from diffusion_hopping.analysis.metrics import (
    MolecularConnectivity,
    MolecularLipinski,
    MolecularLogP,
    MolecularNovelty,
    MolecularQEDValue,
    MolecularSAScore,
    MolecularValidity,
)
from diffusion_hopping.analysis.transform import (
    LargestFragmentTransform,
    UniversalForceFieldTransform,
)


class Evaluator(object):
    def __init__(self, path: Path):
        self.data_module = None
        self.model = None
        self.molecule_builder = MoleculeBuilder(include_invalid=True)
        self.transforms = Compose(
            [LargestFragmentTransform(), UniversalForceFieldTransform()]
        )
        self._output = None
        self.molecular_metrics = None
        self._path = path
        self._metric_columns = []
        self._mode = None

    def reset_output(self):
        self._output = None

    def _setup_molecular_metrics(self):
        self.molecular_metrics = {
            "Novelty": MolecularNovelty(self.data_module.get_train_smiles()),
            "Validity": MolecularValidity(),
            "Connectivity": MolecularConnectivity(),
            "Lipinski": MolecularLipinski(),
            "LogP": MolecularLogP(),
            "QED": MolecularQEDValue(),
            "SAScore": MolecularSAScore(),
        }
        self._metric_columns = list(self.molecular_metrics.keys())

    def load_data_module(self, data_module):
        self.data_module = data_module
        self._setup_molecular_metrics()

    def load_model(self, model):
        self.model = model

    def generate_molecules(
        self, molecules_per_pocket=3, batch_size=32, limit_samples=None
    ):
        self._mode = "sampling"
        self._generate_molecules(
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
            limit_samples=limit_samples,
        )

    def generate_molecules_inpainting(
        self, molecules_per_pocket=3, batch_size=32, limit_samples=None, r=10, j=10
    ):
        self._mode = "inpainting"
        self._generate_molecules(
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
            limit_samples=limit_samples,
            inpaint_scaffold=True,
            r=r,
            j=j,
        )

    def use_ground_truth_molecules(self, limit_samples=None):
        self._mode = "ground_truth"
        self._use_ground_truth_molecules(limit_samples=limit_samples)

    def evaluate(self, transform_for_qvina=True):
        self.enrich_molecule_output()
        self.add_metrics()
        self.store_pockets()
        self.store_molecules(transform=transform_for_qvina)
        self.calculate_qvina_scores()

    def _prepare_dataframe(self, molecules_per_pocket):
        test_loader = self.data_module.test_dataloader()
        test_items = []
        for batch in test_loader:
            test_items.extend(batch.to_data_list())
        test_items, sample_nums = zip(
            *[(item, i) for item in test_items for i in range(molecules_per_pocket)]
        )
        self._output = pd.DataFrame(
            {
                "sample_num": sample_nums,
                "test_set_item": test_items,
            }
        )

        self._output["identifier"] = self._output["test_set_item"].apply(
            lambda x: x.identifier
        )
        self._output = self._output[["identifier", "sample_num", "test_set_item"]]
        self._output = self._output.sort_values(by=["identifier", "sample_num"])

    def _generate_molecules(
        self,
        molecules_per_pocket=3,
        batch_size=32,
        limit_samples=None,
        inpaint_scaffold=False,
        j=10,
        r=10,
    ):
        print("Generating molecules...")
        self.model.eval()
        self.data_module.setup(stage="test")
        self._prepare_dataframe(molecules_per_pocket=molecules_per_pocket)
        if limit_samples is not None:
            self._output = self._output.iloc[:limit_samples]

        device_is_cpu = self.model.device == torch.device("cpu")
        self._output["molecule"] = self._sample_molecules(
            self._output["test_set_item"],
            batch_size,
            inpaint_scaffold,
            j,
            r,
            multi_threading=device_is_cpu,
        )

    def _use_ground_truth_molecules(self, limit_samples=None):
        print("Using ground truth molecules...")
        self.model.eval()
        self.data_module.setup(stage="test")
        self._prepare_dataframe(molecules_per_pocket=1)

        self._output["molecule"] = self._output["test_set_item"].apply(
            lambda x: x["ligand"].ref
        )
        if limit_samples is not None:
            self._output = self._output.iloc[:limit_samples]

    def enrich_molecule_output(self):
        print("Enriching molecule output...")
        self._output["SMILES"] = self._output.apply(_to_smiles, axis=1)
        self._output["Image"] = self._output.apply(self._to_image, axis=1)
        self._output["SMILES-Image"] = self._output.apply(_to_smiles_image, axis=1)

    def add_metrics(self):
        print("Adding metrics...")
        for metric_name, metric in self.molecular_metrics.items():
            self._output[metric_name] = self._output["molecule"].apply(
                lambda x: metric([x]).item()
            )

        self.add_diversity_metric()

    def add_diversity_metric(self):
        if "Diversity" not in self._metric_columns:
            self._metric_columns.append("Diversity")

        self._output["Diversity"] = self._output.groupby("identifier")[
            "molecule"
        ].transform(lambda x: self._calculate_diversity(x))

    def _calculate_diversity(self, x):
        mols = [mol for mol in x if mol is not None]
        if len(mols) == 0:
            return 0.0
        if len(mols) == 1:
            return 1.0

        rdk_fingerprints = [Chem.RDKFingerprint(mol) for mol in mols]

        tanimoto_similarities = [
            DataStructs.TanimotoSimilarity(f1, f2)
            for f1, f2 in itertools.combinations(rdk_fingerprints, 2)
        ]
        return 1 - np.mean(tanimoto_similarities)

    def store_molecules(self, transform=False):
        print("Storing molecules...")
        store_path = self._path / "data"
        self._output["molecule_path"] = self._output.apply(
            lambda row: store_path
            / row["identifier"]
            / f"sample_{row['sample_num']}.pdb"
            if row["molecule"] is not None
            else None,
            axis=1,
        )
        for i, row in tqdm(list(self._output.iterrows())):
            if row["molecule"] is None:
                continue
            self._store_molecule(row["molecule"], row["molecule_path"], transform)

    def _store_molecule(self, mol, path, transform=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if transform:
            mol = self.transforms(mol)
        Chem.MolToPDBFile(
            mol,
            str(path),
        )

    def store_pockets(self):
        print("Storing pockets...")
        store_path = self._path / "data"
        self._output["pocket_path"] = self._output.apply(
            lambda row: store_path / row["identifier"] / "pocket.pdb", axis=1
        )
        for i, row in tqdm(list(self._output.iterrows())):
            pocket_path = row["test_set_item"]["protein"].path
            row["pocket_path"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(pocket_path, str(row["pocket_path"]))

    def calculate_qvina_scores(self):
        print("Calculating QVina scores...")
        scores = thread_map(
            lambda iterrows: qvina_score(iterrows[1]), list(self._output.iterrows())
        )
        self._output["QVina"] = scores
        if "QVina" not in self._metric_columns:
            self._metric_columns.append("QVina")

    def _sample_molecules(
        self,
        items,
        batch_size,
        inpaint_scaffold=False,
        r=10,
        j=10,
        multi_threading=True,
    ):
        loader = DataLoader(list(items), batch_size=batch_size, shuffle=False)
        results_list = []
        if inpaint_scaffold:
            func = functools.partial(self._generate_molecule_inpaint, j=j, r=r)

        else:
            func = self._generate_molecule
        if multi_threading:
            results = thread_map(func, list(loader), desc="Sampling molecules")
            for result in results:
                results_list.extend(result)
        else:
            for batch in tqdm(loader, desc="Sampling molecules"):
                results_list.extend(func(batch))

        return results_list

    @torch.no_grad()
    def _generate_molecule(self, batch):
        batch = batch.to(self.model.device)
        sample_results = self.model.model.sample(batch)
        final_output = sample_results[-1]
        molecules = self.molecule_builder(final_output)
        return molecules

    @torch.no_grad()
    def _generate_molecule_inpaint(self, batch, j=10, r=10) -> List[Chem.Mol]:
        batch = batch.to(self.model.device)
        mask = batch["ligand"].scaffold_mask
        sample_results = self.model.model.inpaint(batch, mask, j=j, r=r)
        final_output = sample_results[-1]
        molecules = self.molecule_builder(final_output)
        return molecules

    def to_html(self, path):
        return to_html(
            self._output.drop(columns=["test_set_item"]),
            path,
            image_columns=["Image", "SMILES-Image"],
        )

    def to_csv(self, path):
        self._output.drop(columns=["test_set_item"]).to_csv(path)

    def to_tensor(self, path):
        torch.save((self._output, self._mode), path)

    def from_tensor(self, path):
        self._output, self._mode = torch.load(path)

    def print_summary_statistics(self):
        print(self.get_summary_string())

    def get_summary_string(self):
        summary_statistics = self.get_summary_statistics()
        summary_string = f"Summary statistics for mode {self._mode}:\n"
        for metric_name, metric_statistics in summary_statistics.items():
            summary_string += f"{metric_name}: {metric_statistics['mean']:.3f} ± {metric_statistics['std']:.3f}\n"
        return summary_string

    def get_summary_statistics(self):
        summary_statistics = {}
        for metric_name in self._metric_columns:
            summary_statistics[metric_name] = {
                "mean": self._output[metric_name].mean(),
                "std": self._output[metric_name].std(),
            }
        return summary_statistics

    def _get_conditional_mask(self, row, mark_scaffold=None):
        if self._mode == "ground_truth":
            return ~row["test_set_item"]["ligand"].scaffold_mask
        elif self._mode == "sampling":
            if mark_scaffold is None:
                return ~self.model.model.get_mask(row["test_set_item"])
            elif mark_scaffold:
                return ~row["test_set_item"]["ligand"].scaffold_mask
            else:
                return torch.ones_like(
                    row["test_set_item"]["ligand"].scaffold_mask
                ).bool()
        elif self._mode == "inpainting":
            return ~row["test_set_item"]["ligand"].scaffold_mask
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def _to_image(self, row):
        mask = self._get_conditional_mask(row)
        atoms_to_highlight = [item.item() for item in torch.where(mask)[0]]

        mol = row["molecule"]
        return _image_with_highlighted_atoms(mol, atoms_to_highlight)

    def is_model_repainting_compatible(self) -> bool:
        return not self.model.model.condition_on_fg

    def output_best_samples(
        self,
        identifier: str,
        sample_nums: List[int],
        n=3,
        transform=True,
        mark_scaffold=True,
    ):
        output = self._output[self._output["identifier"] == identifier]
        output = output[output["sample_num"].isin(sample_nums)]
        output = output.nsmallest(n, "QVina")

        output_path = self._path / "samples" / identifier
        output_path.mkdir(parents=True, exist_ok=True)
        output["molecule_path"] = output.apply(
            lambda row: output_path / f"sample{row['sample_num']}_{self._mode}.pdb",
            axis=1,
        )
        for i, row in output.iterrows():
            self._store_molecule(
                row["molecule"], row["molecule_path"], transform=transform
            )
            qvina_score(row)

        to_html(
            output.drop(columns=["test_set_item"]),
            output_path / f"summary_{self._mode}.html",
            image_columns=["Image", "SMILES-Image"],
        )

        for i, row in output.iterrows():
            image = row["Image"]
            image.save(output_path / f"sample{row['sample_num']}_{self._mode}.png")

            smiles_image = row["SMILES-Image"]
            smiles_image.save(
                output_path / f"sample{row['sample_num']}_{self._mode}_smiles.png"
            )
        for i, row in output.iterrows():
            mol = Chem.Mol(row["molecule"])
            mask = self._get_conditional_mask(row, mark_scaffold=mark_scaffold)
            atoms_to_highlight = [item.item() for item in torch.where(mask)[0]]
            bonds_to_highlight = [
                bond.GetIdx()
                for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() in atoms_to_highlight
                or bond.GetEndAtomIdx() in atoms_to_highlight
            ]
            from rdkit.Chem import rdCoordGen

            rdCoordGen.AddCoords(mol)

            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=atoms_to_highlight,
                highlightBonds=bonds_to_highlight,
            )
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            svg = svg.replace("svg:", "")
            Path(
                output_path / f"sample{row['sample_num']}_{self._mode}_highlight.svg"
            ).write_text(svg)
