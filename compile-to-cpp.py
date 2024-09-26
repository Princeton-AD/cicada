import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import glob
import hls4ml
import numpy as np
import yaml

from drawing import Draw
from hls4ml.model.layers import Activation as ActivationHLS
from hls4ml.model.optimizer import OptimizerPass, register_pass
from huggingface_hub import from_pretrained_keras
from generator import RegionETGenerator
from pathlib import Path
from utils import IsValidFile, CreateFolder


class EliminateLinearActivationCustom(OptimizerPass):
    def match(self, node):
        return (
            isinstance(node, ActivationHLS) and node.get_attr("activation") == "linear"
        )

    def transform(self, model, node):
        model.remove_node(node)
        return True


def get_hls_config(keras_model, version):
    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")

    hls_config["Model"]["Strategy"] = "Latency"

    for layer in hls_config["LayerName"].keys():
        hls_config["LayerName"][layer]["ReuseFactor"] = 2

    hls_config["LayerName"]["inputs_"]["Precision"]["result"] = "ap_uint<10>"
    if version.startswith("1"):
        hls_config["LayerName"]["dense1"]["Precision"]["result"] = "ap_fixed<26, 20>"
        hls_config["LayerName"]["dense1"]["Precision"]["accum"] = "ap_fixed<26, 20>"
    else:
        hls_config["LayerName"]["conv"]["Strategy"] = "Resource"
        hls_config["LayerName"]["conv"]["ReuseFactor"] = 1
        hls_config["LayerName"]["conv"]["ParallelizationFactor"] = 21
        hls_config["LayerName"]["conv"]["Precision"]["result"] = "ap_fixed<30, 22>"
        hls_config["LayerName"]["conv"]["Precision"]["accum"] = "ap_fixed<30, 22>"
        hls_config["LayerName"]["dense1"]["Precision"]["result"] = "ap_fixed<26, 14>"
        hls_config["LayerName"]["dense1"]["Precision"]["accum"] = "ap_fixed<26, 14>"

    hls_config["LayerName"]["dense2"]["Precision"]["result"] = "ap_fixed<26, 14>"
    hls_config["LayerName"]["dense2"]["Precision"]["accum"] = "ap_fixed<26, 14>"

    return hls_config


def convert_to_hls4ml_model(keras_model, hls_config, version="1.0.0"):
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        clock_period=6.25,
        backend="Vitis",
        hls_config=hls_config,
        io_type="io_parallel",
        output_dir="cicada-v{}".format(version),
        part="xc7vx690tffg1927-2",
        project_name="cicada",
        version=version,
    )
    hls_model.compile()
    return hls_model


def testing(keras_model, hls_model, dataset_signals, dataset_background, output_dir, interactive):
    scores = {"scores_hls4ml": {}, "scores_keras": {}}
    for dataset_name, test_vectors in dataset_signals.items():
        test_vectors = test_vectors.reshape(-1, 252)
        scores_hls4ml = hls_model.predict(test_vectors)
        scores_keras = keras_model.predict(test_vectors)
        scores["scores_hls4ml"][dataset_name] = scores_hls4ml.flatten()
        scores["scores_keras"][dataset_name] = scores_keras.flatten()
    test_vectors = dataset_background.reshape(-1, 252)
    scores_hls4ml = hls_model.predict(test_vectors)
    scores_keras = keras_model.predict(test_vectors)
    scores["scores_hls4ml"]["Background"] = scores_hls4ml.flatten()
    scores["scores_keras"]["Background"] = scores_keras.flatten()

    scores_hls4ml = np.concatenate(list(scores["scores_hls4ml"].values()))
    scores_keras = np.concatenate(list(scores["scores_keras"].values()))

    # Generate plots
    draw = Draw(output_dir=output_dir, interactive=interactive)
    name = keras_model.name
    draw.plot_compilation_error(scores_keras, scores_hls4ml, name)
    draw.plot_compilation_error_distribution(scores_keras, scores_hls4ml, name)
    draw.plot_roc_curve_comparison(
        scores["scores_keras"], scores["scores_hls4ml"], name
    )
    draw.plot_cpp_model(hls_model, name)
    draw.plot_output_reference()


def cleanup():
    for f in glob.glob("*.tar.gz"):
        os.remove(f)


def main(args) -> None:

    config = yaml.safe_load(open(args.config))

    # Workaround for linear activation layer removal
    hls4ml.model.flow.flow.update_flow(
        "optimize", remove_optimizers=["eliminate_linear_activation"]
    )
    register_pass(
        "overwrite_eliminate_linear_activation", EliminateLinearActivationCustom
    )
    hls4ml.model.flow.flow.update_flow(
        "convert", add_optimizers=["overwrite_eliminate_linear_activation"]
    )

    # Load QKeras model
    keras_model = from_pretrained_keras(
        "cicada-project/cicada-v{}".format(".".join(args.version.split(".")[:-1]))
    )

    # Genrate hls4ml config
    hls_config = get_hls_config(keras_model, args.version)

    # Genrate hls4ml model
    hls_model = convert_to_hls4ml_model(keras_model, hls_config, args.version)

    # Gather evaluation datasets
    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]
    gen = RegionETGenerator()
    _, _, dataset_background = gen.get_data_split(datasets)
    dataset_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)

    # Final tests of the final configuration
    testing(keras_model, hls_model, dataset_signal, dataset_background, args.output, args.interactive)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Convert QKeras model to hls4ml model"""
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="plots/",
        help="Path to directory where plots will be stored",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "--config",
        "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument("-v", "--version", type=str, help="CICADA version")
    main(parser.parse_args())
