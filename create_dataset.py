import argparse

from _util import get_datamodule
from diffusion_hopping.util import disable_obabel_and_rdkit_logging

if __name__ == "__main__":
    disable_obabel_and_rdkit_logging()
    parser = argparse.ArgumentParser(
        prog="create_dataset.py",
        description="Create dataset with given name",
        epilog="Example: python create_dataset.py crossdocked_filtered",
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    args = parser.parse_args()
    data_module = get_datamodule(args.dataset_name)
    print(
        f"Creating dataset {args.dataset_name} with pre_transform {data_module.pre_transform} and pre_filter {data_module.pre_filter}"
    )
    data_module.setup("fit")
    data_module.setup("test")
