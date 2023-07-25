import argparse
import os
from pathlib import Path

import torch

import wandb
from _util import get_datamodule
from diffusion_hopping.analysis.evaluate import Evaluator
from diffusion_hopping.model import DiffusionHoppingModel
from diffusion_hopping.util import disable_obabel_and_rdkit_logging
from evaluate_model import evaluate_molecules, generate_molecules


def setup_model_and_data_module(
    checkpoint, dataset_name, device="cpu", output_path=None
):
    checkpoint_folder = Path("artifacts") / checkpoint.name
    if not checkpoint_folder.exists():
        checkpoint_folder = checkpoint.download()
    else:
        print("Checkpoint already downloaded")
    checkpoint_path = Path(checkpoint_folder) / "model.ckpt"
    model = DiffusionHoppingModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    ).to(device)

    data_module = get_datamodule(dataset_name, batch_size=32)
    return model, data_module


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_sweep.py",
        description="Evaluate sweep",
        epilog="Example: python evaluate_sweep.py bwgidbfw pdbbind_filtered",
    )
    parser.add_argument(
        "sweep_id",
        type=str,
        help="Sweep id of the sweep to evaluate",
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to evaluate on",
        # choices=get_data_module_choices(),
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to evaluate",
        choices=["ground_truth", "ligand_generation", "inpaint_generation", "all"],
        default="all",
    )
    parser.add_argument(
        "--only_generation",
        action="store_true",
        help="Only generate molecules, do not evaluate them",
    )
    parser.add_argument(
        "--only_evaluation",
        action="store_true",
        help="Only evaluate molecules, do not generate them",
    )
    parser.add_argument(
        "--r",
        type=int,
        help="Number of resampling steps when using inpainting",
        default=10,
    )
    parser.add_argument(
        "--j",
        type=int,
        help="Jump length when using inpainting",
        default=10,
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        help="Limit the number of samples to evaluate",
        default=None,
    )
    parser.add_argument(
        "--molecules_per_pocket",
        type=int,
        help="Number of molecules to generate per pocket",
        default=3,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for generation",
        default=32,
    )
    args = parser.parse_args()

    mode = args.mode
    do_generation = not args.only_evaluation
    do_evaluation = not args.only_generation
    r = args.r
    j = args.j
    limit_samples = args.limit_samples
    molecules_per_pocket = args.molecules_per_pocket
    batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = args.dataset
    sweep_id = args.sweep_id

    api = wandb.Api()
    sweep = api.sweep(f"{os.environ['WANDB_PROJECT']}/{sweep_id}")
    output_path = Path("evaluation") / sweep.name / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    best_run = sweep.best_run()
    artifacts = best_run.logged_artifacts()
    models = [artifact for artifact in artifacts if artifact.type == "model"]
    loss_models = [
        model
        for model in models
        if True or model.metadata["ModelCheckpoint"]["monitor"] == "loss/val"
    ]
    loss_models = sorted(loss_models, key=lambda x: x.metadata["score"])
    artifact = loss_models[0]
    print(f"Using best model: {artifact.name} with score {artifact.metadata['score']}")
    disable_obabel_and_rdkit_logging()

    print("Running on artifact:", artifact.name)

    best_config = best_run.config
    print("Best config:")
    for key, value in best_config.items():
        print(f"> {key}: {value}")

    model, data_module = setup_model_and_data_module(
        artifact, dataset_name, device=device
    )

    evaluator = Evaluator(output_path)
    evaluator.load_data_module(data_module)
    evaluator.load_model(model)

    if do_generation:
        generate_molecules(
            evaluator,
            output_path,
            mode=mode,
            r=r,
            j=j,
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
    if do_evaluation:
        evaluate_molecules(evaluator, output_path, mode=mode)


if __name__ == "__main__":
    main()
