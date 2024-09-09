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
from utils import IsValidFile, CreateFolder


def get_deposits(datasets):
    generator = RegionETGenerator()
    deposits, labels = [], []
    for dataset in datasets:
        name = dataset["name"]
        X = generator.get_data(dataset["path"])
        print(f"{name} samples: {X.shape[0]}")
        deposits.append(X)
        labels.append(name)
    return deposits, labels


def pprint_acceptance(datasets: dict) -> None:
    generator = RegionETGenerator()
    _, acceptance = generator.get_benchmark(datasets, True)
    df = pd.DataFrame(acceptance)
    df["acceptance"] = df["acceptance"].round(decimals=3).astype(str).add(" %")
    print(df.to_markdown(index=False))


def main(args=None) -> None:

    config = yaml.safe_load(open(args.config))

    draw = Draw(output_dir=args.output, interactive=args.interactive)

    for category, dataset in zip(["Background", "Signal"], [config["background"], config["signal"]]):

        deposits, labels = get_deposits(dataset)

        for name, X in zip(labels, deposits):
            draw.plot_regional_deposits(
                np.mean(X, axis=0).reshape(18, 14), np.mean(X, axis=(0, 1, 2, 3)), name
            )

        draw.plot_spacial_deposits_distribution(deposits, labels, name=category, apply_weights=True)
        draw.plot_deposits_distribution(deposits, labels, name=category)

    pprint_acceptance(config["signal"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Profile training and evaluation datasets"""
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="plots/",
        help="Path to directory where plots will be stored",
    )
    parser.add_argument(
        "--config",
        "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    main(parser.parse_args())
