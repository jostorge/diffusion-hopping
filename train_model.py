import argparse
from types import SimpleNamespace

import pytorch_lightning as pl
import torch

import wandb
from _util import get_callbacks, get_datamodule, get_logger, get_model
from diffusion_hopping.model.enum import Architecture
from diffusion_hopping.util import disable_obabel_and_rdkit_logging


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train(config, accelerator="gpu" if torch.cuda.is_available() else None, devices=1):
    run = wandb.init(project="diffusion_hopping", config=config)
    pl.seed_everything(config.seed)

    data_module = get_datamodule(
        config.dataset_name, batch_size=config.batch_size // devices
    )

    model = get_model(
        hidden_features=config.hidden_features,
        num_layers=config.num_layers,
        condition_on_fg=config.condition_on_fg,
        joint_features=config.joint_features,
        architecture=config.architecture,
        attention=config.attention,
        lr=config.lr,
        T=config.T,
        edge_cutoff=config.edge_cutoff,
        ligand_features=data_module.pre_transform.ligand_features,
        protein_features=data_module.pre_transform.protein_features,
    )

    model.setup_metrics(data_module.get_train_smiles())

    wandb_logger = get_logger(run)
    wandb_logger.watch(model)

    callbacks = get_callbacks()
    trainer = pl.Trainer(
        max_steps=config.num_steps,
        accelerator=accelerator,
        devices=devices,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)


def parse_args():
    default_config = SimpleNamespace(
        architecture=Architecture.GVP,
        seed=1,
        dataset_name="pdbbind_filtered",
        condition_on_fg=False,
        num_steps=10000,
        batch_size=32,
        T=500,
        lr=1e-4,
        num_layers=6,
        joint_features=128,
        hidden_features=256,
        edge_cutoff=(None, 5, 5),
    )

    parser = argparse.ArgumentParser(
        prog="train_model.py",
        description="Train model",
        epilog="Example: python train_model.py",
    )
    parser.add_argument(
        "--architecture",
        type=Architecture,
        help="Architecture",
        default=default_config.architecture,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=default_config.seed,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default=default_config.dataset_name,
    )
    parser.add_argument(
        "--condition_on_fg",
        type=str_to_bool,
        help="Condition on functional groups",
        default=default_config.condition_on_fg,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of steps",
        default=default_config.num_steps,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=default_config.batch_size,
    )
    parser.add_argument(
        "--T",
        type=int,
        help="Diffusion time",
        default=default_config.T,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=default_config.lr,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers",
        default=default_config.num_layers,
    )
    parser.add_argument(
        "--joint_features",
        type=int,
        help="Number of joint features",
        default=default_config.joint_features,
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        help="Number of hidden features",
        default=default_config.hidden_features,
    )
    parser.add_argument(
        "--edge_cutoff",
        type=str,
        help="Edge cutoff",
        default=str(default_config.edge_cutoff),
    )
    parser.add_argument(
        "--attention",
        type=str_to_bool,
        help="Use attention",
        default=True,
    )

    config = parser.parse_args()
    config.edge_cutoff = eval(config.edge_cutoff)

    return config


def main():
    disable_obabel_and_rdkit_logging()
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
