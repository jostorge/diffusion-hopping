import argparse
import os
from pathlib import Path

import torch

import wandb
from _util import get_datamodule
from diffusion_hopping.analysis.evaluate import Evaluator
from diffusion_hopping.model import DiffusionHoppingModel
from diffusion_hopping.util import disable_obabel_and_rdkit_logging


def generate_molecules(
    evaluator: Evaluator,
    output_path: Path,
    mode: str = "all",
    r: int = 10,
    j: int = 10,
    limit_samples: int = None,
    molecules_per_pocket: int = 100,
    batch_size: int = 32,
):
    is_repainting_compatible = evaluator.is_model_repainting_compatible()
    if (
        mode == "ground_truth"
        or mode == "all"
        or (mode == "inpaint_generation" and is_repainting_compatible)
    ):
        print("Generating ground truth molecules...")
        evaluator.use_ground_truth_molecules(limit_samples=limit_samples)
        evaluator.to_tensor(output_path / "molecules_ground_truth.pt")

    if mode == "ligand_generation" or mode == "all":
        print("Generating ligand molecules...")
        evaluator.generate_molecules(
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
        evaluator.to_tensor(output_path / "molecules_ligand_generation.pt")

    if mode == "inpaint_generation" or (mode == "all" and is_repainting_compatible):
        print(f"Generating inpaint molecules with r={r}, j={j}...")
        evaluator.generate_molecules_inpainting(
            r=r,
            j=j,
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
        evaluator.to_tensor(output_path / "molecules_inpaint_generation.pt")


def evaluate_molecules(evaluator, output_path, mode="all"):
    is_repainting_compatible = evaluator.is_model_repainting_compatible()
    output_str = f"Output path: {output_path}\n"
    if (
        mode == "ground_truth"
        or mode == "all"
        or (mode == "inpaint_generation" and is_repainting_compatible)
    ):
        print("Running ground truth evaluation...")
        evaluator.from_tensor(output_path / "molecules_ground_truth.pt")
        evaluator.evaluate(transform_for_qvina=False)
        evaluator.to_html(output_path / "results_ground_truth.html")
        evaluator.to_tensor(output_path / "results_ground_truth.pt")
        evaluator.print_summary_statistics()
        output_str += f"Ground truth results: \n{evaluator.get_summary_string()}\n"

    if mode == "ligand_generation" or mode == "all":
        print("Running ligand generation evaluation...")
        evaluator.from_tensor(output_path / "molecules_ligand_generation.pt")
        evaluator.evaluate(transform_for_qvina=True)
        evaluator.to_html(output_path / "results_ligand_generation.html")
        evaluator.to_tensor(output_path / "results_ligand_generation.pt")
        evaluator.print_summary_statistics()
        output_str += f"Ligand generation results: \n{evaluator.get_summary_string()}\n"

    if mode == "inpaint_generation" or (mode == "all" and is_repainting_compatible):
        print("Running inpaint generation evaluation...")
        evaluator.from_tensor(output_path / "molecules_inpaint_generation.pt")
        evaluator.evaluate(transform_for_qvina=True)
        evaluator.to_html(output_path / "results_inpaint_generation.html")
        evaluator.to_tensor(output_path / "results_inpaint_generation.pt")
        evaluator.print_summary_statistics()
        output_str += (
            f"Inpaint generation results: \n{evaluator.get_summary_string()}\n"
        )

    output_path.joinpath("summary.txt").write_text(output_str)


def setup_model_and_data_module(artifact_id, dataset_name, device="cpu"):
    api = wandb.Api()
    checkpoint = api.artifact(artifact_id, type="model")
    checkpoint_folder = checkpoint.download()
    checkpoint_path = Path(checkpoint_folder) / "model.ckpt"

    model = DiffusionHoppingModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    ).to(device)

    data_module = get_datamodule(dataset_name, batch_size=32)
    return model, data_module


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_model.py",
        description="Evaluate ligand generation",
        epilog="Example: python evaluate_model.py bwgidbfw pdbbind_filtered",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run id of the model to evaluate",
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
    run_id = args.run_id
    r = args.r
    j = args.j
    limit_samples = args.limit_samples
    molecules_per_pocket = args.molecules_per_pocket
    batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    artifact_id = f"{os.environ['WANDB_PROJECT']}/model-{run_id}:best_k"

    dataset_name = args.dataset
    output_path = Path("evaluation") / run_id / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    disable_obabel_and_rdkit_logging()

    print("Running on artifact:", artifact_id)

    model, data_module = setup_model_and_data_module(
        artifact_id, dataset_name, device=device
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
