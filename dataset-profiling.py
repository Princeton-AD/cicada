import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import pandas as pd
import yaml

from drawing import Draw
from generator import RegionETGenerator
from pathlib import Path
from typing import List
from utils import IsValidFile


def dataset_profiling(datasets: dict, prefix: str) -> None:
    draw = Draw()
    generator = RegionETGenerator()
    deposits, labels = [], []
    for dataset in datasets:
        name = dataset["name"]
        X = generator.get_data(dataset["path"])
        print(f"{name} samples: {X.shape[0]}")
        draw.plot_regional_deposits(
            np.mean(X, axis=0).reshape(18, 14), np.mean(X, axis=(0, 1, 2, 3)), name
        )
        deposits.append(X)
        labels.append(name)
    draw.plot_spacial_deposits_distribution(deposits, labels, name=f"{prefix}")
    draw.plot_deposits_distribution(deposits, labels, name=f"{prefix}")


def pprint_acceptance(datasets: dict) -> None:
    generator = RegionETGenerator()
    _, acceptance = generator.get_benchmark(datasets, True)
    df = pd.DataFrame(acceptance)
    df["acceptance"] = df["acceptance"].round(decimals=3).astype(str).add(" %")
    print(df.to_markdown(index=False))


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description="""Profile training and evaluation datasets"""
    )
    parser.add_argument(
        "--config",
        "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    args = parser.parse_args()
    return yaml.safe_load(open(args.config))


def main(args_in=None) -> None:
    config = parse_arguments()
    dataset_profiling(config["background"], "Background")
    dataset_profiling(config["signal"], "Signal")
    pprint_acceptance(config["signal"])


if __name__ == "__main__":
    main()
