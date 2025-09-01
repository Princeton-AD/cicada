import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # because gpu:0 is being used by another heavy process, so not enough memory on it
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
#from models import TeacherAutoencoder, CicadaV1, CicadaV2
from hgq2_models import TeacherAutoencoder, cicadav2_hgq2
from hgq.utils.sugar import FreeEBOPs
import keras
import gc

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
    #X_hat = X_hat_dict['teacher_outputs'] # needed because we are loading a old model into keras3
    y = loss(X, X_hat)
    y = quantize(np.log(y) * 32)
    dataset = gen.get_generator(X.reshape((-1, 252, 1)), y, 1024, True)#model has reshape
    #dataset = gen.get_generator(X.reshape((-1, 18, 14, 1)), y, 1024, True)# no reshape
    # fixing memory leak
    del X_hat
    del y
    gc.collect()
    return dataset


def train_model(
    model: Model,
    gen_train: data.Dataset,
    gen_val: data.Dataset,
    epoch: int = 1,
    steps: int = 1,
    callbacks=None,
    verbose: bool = False,
    tag = 'teacher',
) -> None:
    print(f"training model {tag}")
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
    print(X_train.shape)
    print(X_val.shape)#; input("got shapes?")
    #X_signal, _ = gen.get_benchmark(config["signal"], filter_acceptance=False)
    gen_train = gen.get_generator(X_train, X_train, 512, True)
    print(len(gen_train))#1114
    gen_val = gen.get_generator(X_val, X_val, 512)
    #outlier_train = gen.get_data(config["exposure"]["training"])
    #outlier_val = gen.get_data(config["exposure"]["validation"])
    outlier_train, outlier_val = gen.generate_random_exposure_data(X_train,X_val,500_000,100_000)

    print(outlier_train.shape)
    print(outlier_val.shape)#; input("got outlier shapes?")

    X_train_student = np.concatenate([X_train, outlier_train])
    X_val_student = np.concatenate([X_val, outlier_val])

    teacher = TeacherAutoencoder((18, 14, 1)).get_model()
    teacher.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model_out_dir = f"{args.output}/{teacher.name}"
    os.makedirs(model_out_dir,exist_ok=True)
    checkpoint_file = os.path.join(model_out_dir, "model_checkpoint.keras")
    training_log = os.path.join(model_out_dir, "training.log")    
    t_mc = ModelCheckpoint(checkpoint_file, save_best_only=True)
    t_log = CSVLogger(training_log, append=True)
    #teacher_layer = keras.layers.TFSMLayer("models_rand_1/teacher", call_endpoint='serving_default') # using pretrained teacher
    #input_shape = (18, 14, 1) 
    #inputs = keras.Input(shape=input_shape)
    #outputs = teacher_layer(inputs)
    #teacher = keras.Model(inputs, outputs) # needed because we are loading a old model into keras3
    input_shape = (252,)

    student_hgq = cicadav2_hgq2(input_shape).get_model()
    student_hgq.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    model_out_dir = f"{args.output}/{student_hgq.name}"
    os.makedirs(model_out_dir,exist_ok=True)
    checkpoint_file = os.path.join(model_out_dir, "model_checkpoint.keras")
    training_log = os.path.join(model_out_dir, "training.log")    
    student_hgq_mc = ModelCheckpoint(checkpoint_file, save_best_only=True)
    student_hgq_log = CSVLogger(training_log, append=True)
    ebops = FreeEBOPs()

    #cicada_v1 = CicadaV1((252,)).get_model()
    #cicada_v1.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    #cv1_mc = ModelCheckpoint(f"{args.output}/{cicada_v1.name}", save_best_only=True)
    #cv1_log = CSVLogger(f"{args.output}/{cicada_v1.name}/training.log", append=True)

    #cicada_v2 = CicadaV2((252,)).get_model()
    #cicada_v2.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    #cv2_mc = ModelCheckpoint(f"{args.output}/{cicada_v2.name}", save_best_only=True)
    #cv2_log = CSVLogger(f"{args.output}/{cicada_v2.name}/training.log", append=True)

    for epoch in range(args.epochs):
        train_model(
            teacher,
            gen_train,
            gen_val,
            epoch=epoch,
            callbacks=[t_mc, t_log],
            verbose=args.verbose,
        )

        tmp_teacher = load_model(f"{args.output}/teacher/model_checkpoint.keras")#teacher#
        s_gen_train = get_student_targets(tmp_teacher, gen, X_train_student)
        s_gen_val = get_student_targets(tmp_teacher, gen, X_val_student)
        
        train_model(
            student_hgq,
            s_gen_train,
            s_gen_val,
            epoch=10 * epoch,
            steps=10,
            callbacks=[student_hgq_mc, ebops, student_hgq_log],
            verbose=args.verbose,
            tag='v2'
        )         

        '''
        train_model(
            cicada_v1,
            s_gen_train,
            s_gen_val,
            epoch=10 * epoch,
            steps=10,
            callbacks=[cv1_mc, cv1_log],
            verbose=args.verbose,
            tag='v1'
        )
        train_model(
            cicada_v2,
            s_gen_train,
            s_gen_val,
            epoch=10 * epoch,
            steps=10,
            callbacks=[cv2_mc, cv2_log],
            verbose=args.verbose,
            tag='v2'
        )
        '''
        # fixing memory leak
        del s_gen_train
        del s_gen_val
        del tmp_teacher
        gc.collect()


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
        default="models_rand_hgq2_epochs10_max1k_uniform_with_reshape_teacher_and_student/",
        help="Path to directory where models will be stored",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs",
        default=10,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=True,
    )
    main(parser.parse_args())
