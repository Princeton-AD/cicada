import argparse
import os
import keras
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple

class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # folder_name = folder_name.rstrip(os.sep)
        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


def predict_single_image(model, image):
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image, verbose=False)['teacher_outputs']
    return np.squeeze(pred, axis=0)


def get_fractions_above_threshold(scores):
    thresholds = scores.flatten()
    thresholds.sort()
    fractions = np.linspace(1, 0, len(thresholds))
    return thresholds, fractions

def quantize(arr: npt.NDArray, precision: tuple = (16, 8)) -> npt.NDArray:
    word, int_ = precision
    decimal = word - int_
    step = 1 / 2**decimal
    max_ = 2**int_ - step
    arrq = step * np.round(arr / step)
    arrc = np.clip(arrq, 0, max_)
    return arrc

def loss(y_true: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
        loss = np.mean((y_true - y_pred) ** 2, axis=(1, 2, 3))   
        return loss    


def get_teacher_model() -> keras.Model:
    """Loads the pre-trained teacher model using TFSMLayer for Keras 3 compatibility."""
    print("Loading teacher model...")
    teacher_model_dir = "models_rand_1/teacher" 
    if not os.path.exists(teacher_model_dir):
        print(f"Warning: Teacher model directory not found at '{teacher_model_dir}'. Creating a dummy model.")
        input_shape = (18, 14, 1)
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
        outputs = keras.layers.Reshape(input_shape)(outputs)
        return keras.Model(inputs, outputs)

    teacher_layer = keras.layers.TFSMLayer(teacher_model_dir, call_endpoint='serving_default')
    input_shape = (18, 14, 1)
    inputs = keras.Input(shape=input_shape)
    # The TFSMLayer output is a dictionary, so we need to specify the key
    outputs = teacher_layer(inputs)['teacher_outputs']
    teacher = keras.Model(inputs, outputs)
    return teacher

def get_qkeras_student_model(model_path) -> keras.Model:
    """Loads the qkeras student model using TFSMLayer for Keras 3 compatibility."""
    print("Loading qkeras model...")
    if not os.path.exists(model_path):
        print(f"Warning: Qkeras model directory not found at '{model_path}'. Creating a dummy model.")
        input("what to do?")
    qkeras_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    input_shape = (252,)
    inputs = keras.Input(shape=input_shape)
    # The TFSMLayer output is a dictionary, so we need to specify the key
    outputs = qkeras_layer(inputs)['outputs']
    qkeras_student = keras.Model(inputs, outputs)
    return qkeras_student 

def get_ebops_from_log(log_path: str) -> float:
    """Parses a log file to find the ebops at the best validation loss."""
    try:
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found at {log_path}")
            return np.nan
        df = pd.read_csv(log_path)    
        precision = 5
        df['rounded_val_loss'] = df['val_loss'].round(decimals=precision)
        min_rounded_loss = df['rounded_val_loss'].min()
        # Get all rows from the original dataframe where the rounded loss matches the minimum
        best_rows = df[df['rounded_val_loss'] == min_rounded_loss]
        best_epoch_row = best_rows.loc[best_rows.index.max()]
        return best_epoch_row['ebops']
    except Exception as e:
        print(f"Error processing log file {log_path}: {e}")
        return np.nan

def get_student_models() -> Dict:
    """Loads all student models and their corresponding log file paths."""
    from hgq.layers import QConv2D, QDense, QBatchNormDense
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from tensorflow.keras.models import load_model
    print("Loading student models...")
    student_model_string_20 = 'models_epochs20_with_reshape_hgq_old_Sig_exposure_1to1_beta%s_onlystudent_savebestTrue'
    student_model_string_100 = 'models_epochs100_with_reshape_hgq_old_Sig_exposure_1to1_beta%s_onlystudent_savebestTrue'
    student_test = "models_epochs100_with_reshape_hgq_tt2l2nu_traintestvalsplit_exposure_weighted1to1_beta%s_onlystudent_savebestTrue_using_models_rand_1_for_training"
    student_model_string = student_model_string_20#student_test#student_model_string_100
    student_model_dict = {
        '1Em3': {'folder': student_model_string % '1Em3'},
        '1Em4': {'folder': student_model_string % '1Em4'},
        '1Em5': {'folder': student_model_string % '1Em5'},
        '1Em6': {'folder': student_model_string % '1Em6'},
    }
    custom_objects_dict = {
        "QuantizerConfigScope": QuantizerConfigScope,
        "LayerConfigScope": LayerConfigScope,
        "QConv2D": QConv2D,
        "QBatchNormDense": QBatchNormDense,
        "QDense": QDense,
    }
    
    loaded_students = {}
    for key, info in student_model_dict.items():
        model_path = os.path.join(info['folder'], 'hgq2_model', 'model_checkpoint.keras')
        log_path = os.path.join(info['folder'], 'hgq2_model', 'training.log')
        if os.path.exists(model_path) and os.path.exists(log_path):
            print(f"  Loading student '{key}' from {model_path}")
            loaded_students[key] = {
                'model': load_model(model_path, custom_objects=custom_objects_dict),
                'name': key,
                'log_path': log_path
            }
        else:
            print(f"Warning: Could not find model for student '{key}'. Searched at '{model_path}'. Skipping.")
    #load qkeras model
    qkeras_base = '/nfs_scratch/mallampalli/open_data/cicada/training/cicada/'
    qkeras_20 = qkeras_base + 'models_only_student_qkeras_old_Sig_exposure_20epochs' 
    qkeras_100 =  qkeras_base + 'models_only_student_qkeras_old_Sig_exposure_100epochs'   
    qkeras_test = qkeras_base + 'models_only_student_qkeras_tt2l2nu_traintestvalsplit_exposure_weighted1to1_100epochs'
    qkeras_path =  qkeras_20#qkeras_test#qkeras_100
    model_path = os.path.join(qkeras_path, 'cicada-v2')
    log_path = os.path.join(qkeras_path, 'cicada-v2', 'training.log')
    loaded_students['qkeras'] = {
                'model': get_qkeras_student_model(model_path),
                'name': 'qkeras',
                'log_path': log_path
            } 
    return loaded_students        