import argparse
import os
from pathlib import Path

import pytorch_lightning as pl

import wandb
from _util import get_callbacks, get_datamodule, get_logger
from diffusion_hopping.model.lightning import DiffusionHoppingModel


def guess_artifact_id(run_id: str) -> str:
    return f"{os.environ['WANDB_PROJECT']}/model-{run_id}:best_k"


def resume_training(
    run_id: str,
    artifact_id: str,
    devices: int = 1,
):
    run = wandb.init(project="diffusion_hopping", id=run_id, resume="must")

    wandb_logger = get_logger(run)

    batch_size = wandb_logger.experiment.config["batch_size"]
    dataset_name = wandb_logger.experiment.config["dataset_name"]

    checkpoint = wandb_logger.experiment.use_artifact(artifact_id, type="model")
    checkpoint_path = Path(checkpoint.download()) / "model.ckpt"

    model = DiffusionHoppingModel.load_from_checkpoint(checkpoint_path)
    data_module = get_datamodule(dataset_name, batch_size=batch_size // devices)
    model.setup_metrics(data_module.get_train_smiles())
    wandb_logger.watch(model)

    root_dir = Path(os.getcwd()) / run_id
    root_dir.mkdir(exist_ok=True, parents=True)
    callbacks = get_callbacks()

    trainer = pl.Trainer(
        max_steps=wandb_logger.experiment.config["num_steps"],
        accelerator="gpu",
        devices=devices,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=root_dir,
    )
    checkpoint_path = "hpc"

    trainer.fit(
        model,
        data_module,
        ckpt_path=str(checkpoint_path),
    )


def main():

    parser = argparse.ArgumentParser(
        prog="resume_training.py",
        description="Resume training",
        epilog="Example: python resume_training.py bwgidbfw",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run id of the model to resume training",
    )
    parser.add_argument(
        "--artifact_id",
        type=str,
        help="Artifact id of the model to resume training",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=1,
    )
    args = parser.parse_args()

    artifact_id = (
        guess_artifact_id(args.run_id) if args.artifact_id is None else args.artifact_id
    )
    pl.seed_everything(args.seed)
    resume_training(args.run_id, artifact_id)


if __name__ == "__main__":
    main()
