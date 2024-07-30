import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import numpy.typing as npt
import yaml

from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

from utils import IsValidFile, CreateFolder
from generator import RegionETGenerator
from models import TeacherAutoencoder, CicadaV1, CicadaV2


def loss(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
    return np.mean((y_true - y_pred) ** 2, axis=(1, 2, 3))


def quantize(arr: npt.NDArray, precision: tuple = (16, 8)) -> npt.NDArray:
    word, int_ = precision
    decimal = word - int_
    step = 1 / 2**decimal
    max_ = 2**int_ - step
    arrq = step * np.round(arr / step)
    arrc = np.clip(arrq, 0, max_)
    return arrc


def get_student_targets(
    teacher: Model, gen: RegionETGenerator, X: npt.NDArray
) -> data.Dataset:
    X_hat = teacher.predict(X, batch_size=512, verbose=0)
    y = loss(X, X_hat)
    y = quantize(np.log(y) * 32)
    return gen.get_generator(X.reshape((-1, 252, 1)), y, 1024, True)


def train_model(
    model: Model,
    gen_train: data.Dataset,
    gen_val: data.Dataset,
    epoch: int = 1,
    steps: int = 1,
    callbacks=None,
    verbose: bool = False,
) -> None:
    model.fit(
        gen_train,
        steps_per_epoch=len(gen_train),
        initial_epoch=epoch,
        epochs=epoch + steps,
        validation_data=gen_val,
        callbacks=callbacks,
        verbose=verbose,
    )


def main(args) -> None:

    config = yaml.safe_load(open(args.config))

    datasets = [i["path"] for i in config["background"] if i["use"]]
    datasets = [path for paths in datasets for path in paths]

    gen = RegionETGenerator()
    X_train, X_val, X_test = gen.get_data_split(datasets)
    X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)
    gen_train = gen.get_generator(X_train, X_train, 512, True)
    gen_val = gen.get_generator(X_val, X_val, 512)
    outlier_train = gen.get_data(config["exposure"]["training"])
    outlier_val = gen.get_data(config["exposure"]["validation"])

    X_train_student = np.concatenate([X_train, outlier_train])
    X_val_student = np.concatenate([X_val, outlier_train])

    teacher = TeacherAutoencoder((18, 14, 1)).get_model()
    teacher.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    t_mc = ModelCheckpoint(f"{args.output}/{teacher.name}", save_best_only=True)
    t_log = CSVLogger(f"{args.output}/{teacher.name}/training.log", append=True)

    cicada_v1 = CicadaV1((252,)).get_model()
    cicada_v1.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    cv1_mc = ModelCheckpoint(f"{args.output}/{cicada_v1.name}", save_best_only=True)
    cv1_log = CSVLogger(f"{args.output}/{cicada_v1.name}/training.log", append=True)

    cicada_v2 = CicadaV2((252,)).get_model()
    cicada_v2.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    cv2_mc = ModelCheckpoint(f"{args.output}/{cicada_v2.name}", save_best_only=True)
    cv2_log = CSVLogger(f"{args.output}/{cicada_v2.name}/training.log", append=True)

    for epoch in range(args.epochs):
        train_model(
            teacher,
            gen_train,
            gen_val,
            epoch=epoch,
            callbacks=[t_mc, t_log],
            verbose=args.verbose,
        )

        tmp_teacher = load_model(f"{args.output}/teacher")
        s_gen_train = get_student_targets(tmp_teacher, gen, X_train_student)
        s_gen_val = get_student_targets(tmp_teacher, gen, X_val_student)

        train_model(
            cicada_v1,
            s_gen_train,
            s_gen_val,
            epoch=10 * epoch,
            steps=10,
            callbacks=[cv1_mc, cv1_log],
            verbose=args.verbose,
        )
        train_model(
            cicada_v2,
            s_gen_train,
            s_gen_val,
            epoch=10 * epoch,
            steps=10,
            callbacks=[cv2_mc, cv2_log],
            verbose=args.verbose,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""CICADA training scripts""")
    parser.add_argument(
        "--config", "-c",
        action=IsValidFile,
        type=Path,
        default="misc/config.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="models/",
        help="Path to directory where models will be stored",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs",
        default=100,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
