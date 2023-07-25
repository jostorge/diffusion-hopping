from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from diffusion_hopping.data.dataset import CrossDockedDataModule, PDBBindDataModule
from diffusion_hopping.data.featurization import ProteinLigandSimpleFeaturization
from diffusion_hopping.data.filter import QEDThresholdFilter
from diffusion_hopping.model import DiffusionHoppingModel
from diffusion_hopping.model.enum import Architecture, Parametrization


def get_data_module_choices():
    choices = ["pdbbind", "crossdocked"]
    choices += [f"{c}_filtered" for c in choices]
    choices += [f"{c}_full" for c in choices]
    return choices


def get_datamodule(dataset_name: str, batch_size: int = 32):
    if dataset_name not in get_data_module_choices():
        raise ValueError(f"Unknown dataset name {dataset_name}")
    """Create dataset with given name, e.g. crossdocked_filtered or pdbbind_filtered_full"""
    dataset_parts = dataset_name.split("_")
    if dataset_parts[0] == "crossdocked":
        dataset_constructor = CrossDockedDataModule
    elif dataset_parts[0] == "pdbbind":
        dataset_constructor = PDBBindDataModule
    else:
        raise ValueError("Unknown dataset name")

    if len(dataset_parts) == 1:
        pre_transform = ProteinLigandSimpleFeaturization(
            c_alpha_only=True, cutoff=8.0, mode="residue"
        )
        pre_filter = None
    elif len(dataset_parts) == 2:
        if dataset_parts[1] == "filtered":
            pre_transform = ProteinLigandSimpleFeaturization(
                c_alpha_only=True, cutoff=8.0, mode="residue"
            )
            pre_filter = QEDThresholdFilter(0.3)
        elif dataset_parts[1] == "full":
            pre_transform = ProteinLigandSimpleFeaturization(
                c_alpha_only=False, cutoff=8.0, mode="residue"
            )
            pre_filter = None
        else:
            raise ValueError("Unknown dataset name")
    elif len(dataset_parts) == 3:
        if dataset_parts[1] == "filtered" and dataset_parts[2] == "full":
            pre_transform = ProteinLigandSimpleFeaturization(
                c_alpha_only=False, cutoff=8.0, mode="residue"
            )
            pre_filter = QEDThresholdFilter(0.3)
        else:
            raise ValueError("Unknown dataset name")
    else:
        raise ValueError("Unknown dataset name")

    dataset = dataset_constructor(
        f"data/{dataset_name}/",
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        batch_size=batch_size,
        val_batch_size=32,
        test_batch_size=32,
    )
    return dataset


def get_logger(run, **kwargs):
    return WandbLogger(log_model="all", experiment=run, **kwargs)


def get_callbacks():
    val_checkpoint = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={loss/val:.3f}",
        monitor="loss/val",
        mode="min",
        auto_insert_metric_name=False,
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="epoch",
        mode="max",
        every_n_train_steps=25000,
        save_top_k=-1,
    )
    return [val_checkpoint, latest_checkpoint]


def get_model(
    hidden_features=256,
    num_layers=6,
    joint_features=128,
    condition_on_fg=True,
    architecture=Architecture.EGNN,
    lr=1e-4,
    T=1000,
    edge_cutoff=(None, 5, 5),
    ligand_features=10,
    protein_features=20,
    attention=False,
):
    return DiffusionHoppingModel(
        T=T,
        parametrization=Parametrization.EPS,
        lr=lr,
        clip_grad=True,
        condition_on_fg=condition_on_fg,
        x_norm=4.0,
        architecture=architecture,
        edge_cutoff=edge_cutoff,
        hidden_features=hidden_features,
        joint_features=joint_features,
        num_layers=num_layers,
        ligand_features=ligand_features,
        protein_features=protein_features,
        attention=attention,
    )
