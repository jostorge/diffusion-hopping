import os
import sys
from typing import List

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, RDConfig
from torchmetrics import Metric

from diffusion_hopping.analysis.util import largest_component

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


class MolecularValidity(Metric):
    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("num_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        self.num_valid += sum(1 for mol in molecules if mol is not None)
        self.num_total += len(molecules)

    def compute(self):
        return self.num_valid / self.num_total


class MolecularConnectivity(Metric):
    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("num_connected", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        largest_components = largest_component(molecules)

        self.num_total += len(molecules)
        self.num_connected += sum(
            1
            for mol, ref in zip(largest_components, molecules)
            if mol.GetNumAtoms() == ref.GetNumAtoms()
        )

    def compute(self):
        return self.num_connected / self.num_total


class MolecularUniqueness(Metric):
    # TODO fix this metric somehow to work with distributed training in the torchmetrics framework
    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.smiles = []

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        self.smiles.extend(smiles)

    def compute(self):
        return len(set(self.smiles)) / max(len(self.smiles), 1)

    def reset(self) -> None:
        self.smiles = []
        return super().reset()


class MolecularNovelty(Metric):
    higher_is_better = True

    def __init__(self, original_smiles: List[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.original_smiles = set(original_smiles)
        self.add_state("num_novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):

        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        self.num_novel += sum(
            1 for smile in smiles if smile not in self.original_smiles
        )

    def compute(self):
        return self.num_novel / self.num_total


class MolecularQEDValue(Metric):
    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("qed_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        self.qed_sum += sum(Descriptors.qed(mol) for mol in molecules)

    def compute(self):
        return self.qed_sum / self.num_total


class MolecularSAScore(Metric):
    higher_is_better = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("sa_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        # normalize to [0, 1], where 1 is the best score
        self.sa_sum += sum((10 - sascorer.calculateScore(mol)) / 9 for mol in molecules)

    def compute(self):
        return self.sa_sum / self.num_total


class MolecularLogP(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("logp_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        self.logp_sum += sum(Crippen.MolLogP(mol) for mol in molecules)

    def compute(self):
        return self.logp_sum / self.num_total


class MolecularLipinski(Metric):
    higher_is_better = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("lipinski_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, molecules: List[Chem.Mol]):
        molecules = [mol for mol in molecules if mol is not None]
        molecules = largest_component(molecules)

        self.num_total += len(molecules)
        self.lipinski_sum += sum(self._lipinski_score(mol) for mol in molecules)

    def _lipinski_score(self, mol: Chem.Mol) -> int:
        logp = Crippen.MolLogP(mol)
        value = 0
        if Descriptors.ExactMolWt(mol) < 500:
            value += 1
        if Lipinski.NumHDonors(mol) <= 5:
            value += 1
        if Lipinski.NumHAcceptors(mol) <= 10:
            value += 1
        if -2 <= logp <= 5:
            value += 1
        if Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10:
            value += 1
        return value

    def compute(self):
        return self.lipinski_sum / self.num_total
