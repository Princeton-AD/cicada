from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    Conv2DTranspose
)
from hgq.layers import QConv2D, QDense, QBatchNormDense
from hgq.config import LayerConfigScope, QuantizerConfigScope

class TeacherAutoencoder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher")

class cicadav2_hgq2:
    def __init__(self, input_shape: tuple, beta: float=1e-4):
        self.input_shape = input_shape
        self.beta = beta
    def get_model(self): 
        print(f"Using{self.beta}")
        with QuantizerConfigScope(q_type='kif', place='weight', overflow_mode='SAT_SYM', round_mode='RND'):
                with QuantizerConfigScope(q_type='kif', place='datalane', overflow_mode='SAT_SYM', round_mode='RND'):
                        with LayerConfigScope(enable_ebops=True, beta0=self.beta):

                                model = Sequential([
                                        Input(shape=self.input_shape, name="inputs_"),
                                        Reshape((18, 14, 1), name="reshape"),
                                        QConv2D(4, (2, 2), strides=2,padding="valid",activation='relu',use_bias=False,name="conv"),
                                        Flatten(name="flatten"),
                                        Dropout(1 / 9),
                                        QBatchNormDense(16,activation='relu',name="dense1"),
                                        Dropout(1 / 8),
                                        QDense(1,use_bias=False,activation='relu',name="dense2")
                                ],name="hgq2_model")
        return model

if __name__ == "__main__":
        updated_model = hgq_model((252,)).get_model()
        updated_model.compile(optimizer="adam", loss="mae")        
        updated_model.summary()