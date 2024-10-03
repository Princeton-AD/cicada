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
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits


class TeacherAutoencoder:
    def __init__(self, input_shape: tuple, Lambda = [0., 0.], filters = [20, 30, 80], pooling = (2, 2), search=False, compile=False, name="teacher"):
        self.input_shape = input_shape
        self.name = name
        self.Lambda = Lambda
        self.filters = filters
        self.pooling = pooling
        self.search = search
        self.compile = compile

    def get_model(self, hp):
        if not self.search:
            l2_reg_en = regularizers.l2(self.Lambda[0])
            l2_reg_de = regularizers.l2(self.Lambda[1])
            filters = self.filters
            pooling = self.pooling
        else:
            l2_reg_en = regularizers.l2(hp.Float("l2_reg_en", min_value=1e-5, max_value=1., step=10, sampling="log"))
            l2_reg_de = regularizers.l2(hp.Float("l2_reg_de", min_value=1e-5, max_value=1., step=10, sampling="log"))
            filters = [hp.Int("filter_1", min_value=10, max_value=30, step=5),
                       hp.Int("filter_2", min_value=20, max_value=40, step=5),
                       hp.Int("filter_3", min_value=60, max_value=100, step=5)]
            pooling = (hp.Choice("pool_1", [1,2,3]), hp.Choice("pool_2", [1,2,3]))

        inputs = Input(shape=self.input_shape, name=f"{self.name}_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D(pooling, name="teacher_pool_1")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(filters[2], activation="relu", kernel_regularizer=l2_reg_en, name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, kernel_regularizer=l2_reg_de, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name=f"{self.name}_outputs",
        )(x)
        model = Model(inputs, outputs, name=self.name)
        if not self.compile: return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=[MeanSquaredError()])
        return model

class TeacherAutoencoderRevised:
    def __init__(self, input_shape: tuple, Lambda = [0., 0.], filters = [20, 30, 80], pooling = (2, 2), search=False, compile=False, name="teacher-transpose"):
        self.input_shape = input_shape
        self.name = name
        self.Lambda = Lambda
        self.filters = filters
        self.pooling = pooling
        self.search = search
        self.compile = compile

    def get_model(self, hp):
        if not self.search:
            l2_reg_en = regularizers.l2(self.Lambda[0])
            l2_reg_de = regularizers.l2(self.Lambda[1])
            filters = self.filters
            pooling = self.pooling
        else:
            l2_reg_en = regularizers.l2(hp.Float("l2_reg_en", min_value=1e-5, max_value=1., step=10, sampling="log"))
            l2_reg_de = regularizers.l2(hp.Float("l2_reg_de", min_value=1e-5, max_value=1., step=10, sampling="log"))
            filters = [hp.Int("filter_1", min_value=10, max_value=30, step=5),
                       hp.Int("filter_2", min_value=20, max_value=40, step=5),
                       hp.Int("filter_3", min_value=60, max_value=100, step=5)]
            pooling = (hp.Choice("pool_1", [1,2,3]), hp.Choice("pool_2", [1,2,3]))

        inputs = Input(shape=self.input_shape, name=f"{self.name}_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D(pooling, name="teacher_pool_1")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(filters[2], activation="relu", kernel_regularizer=l2_reg_en, name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, kernel_regularizer=l2_reg_de, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = Conv2DTranspose(30, (3, 3), strides=2, padding="same", name="teacher_conv_transpose")(x)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name=f"{self.name}_outputs",
        )(x)
        model = Model(inputs, outputs, name=self.name)
        if not self.compile: return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=[MeanSquaredError()])
        return model

class TeacherScnAutoencoder:
    def __init__(self, input_shape: tuple, Lambda = [0., 0.], filters = [8, 12, 30], pooling=(2, 2), search=False, compile=False, name="teacher_scn"):
        self.input_shape = input_shape
        self.name = name
        self.Lambda = Lambda
        self.filters = filters
        self.pooling = pooling
        self.search = search
        self.compile = compile

    def get_model(self, hp):
        if not self.search:
            l2_reg_en = regularizers.l2(self.Lambda[0])
            l2_reg_de = regularizers.l2(self.Lambda[1])
            filters = self.filters
            pooling = self.pooling
        else:
            l2_reg_en = regularizers.l2(hp.Float("l2_reg_en", min_value=1e-5, max_value=1., step=10, sampling="log"))
            l2_reg_de = regularizers.l2(hp.Float("l2_reg_de", min_value=1e-5, max_value=1., step=10, sampling="log"))
            filters = [hp.Int("filter_1", min_value=4, max_value=10, step=1),
                       hp.Int("filter_2", min_value=6, max_value=14, step=1),
                       hp.Int("filter_3", min_value=16, max_value=32, step=2)]
            pooling = (hp.Choice("pool_1", [1,2,3]), hp.Choice("pool_2", [1,2,3]))

        inputs = Input(shape=self.input_shape, name=f"{self.name}_inputs_")
        x = Reshape((6, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D(pooling, name="teacher_pool_1")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(filters[2], activation="relu", kernel_regularizer=l2_reg_en, name="teacher_latent")(x)
        x = Dense(3 * 7 * 30, kernel_regularizer=l2_reg_de, name="teacher_dense")(x)
        x = Reshape((3, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name=f"{self.name}_outputs",
        )(x)
        model = Model(inputs, outputs, name=self.name)
        if not self.compile: return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=[MeanSquaredError()])
        return model

class TeacherScnAutoencoderRevised:
    def __init__(self, input_shape: tuple, Lambda = [0., 0.], filters = [8, 12, 30], pooling=(2, 2), search=False, compile=False, name="teacher_scn_revised"):
        self.input_shape = input_shape
        self.name = name
        self.Lambda = Lambda
        self.filters = filters
        self.pooling = pooling
        self.search = search
        self.compile = compile

    def get_model(self, hp):
        if not self.search:
            l2_reg_en = regularizers.l2(self.Lambda[0])
            l2_reg_de = regularizers.l2(self.Lambda[1])
            filters = self.filters
            pooling = self.pooling
        else:
            l2_reg_en = regularizers.l2(hp.Float("l2_reg_en", min_value=1e-5, max_value=1., step=10, sampling="log"))
            l2_reg_de = regularizers.l2(hp.Float("l2_reg_de", min_value=1e-5, max_value=1., step=10, sampling="log"))
            filters = [hp.Int("filter_1", min_value=4, max_value=10, step=1),
                       hp.Int("filter_2", min_value=6, max_value=14, step=1),
                       hp.Int("filter_3", min_value=16, max_value=32, step=2)]
            pooling = (hp.Choice("pool_1", [1,2,3]), hp.Choice("pool_2", [1,2,3]))

        inputs = Input(shape=self.input_shape, name=f"{self.name}_inputs_")
        x = Reshape((6, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D(pooling, name="teacher_pool_1")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_en, name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(filters[2], activation="relu", kernel_regularizer=l2_reg_en, name="teacher_latent")(x)
        x = Dense(3 * 7 * 30, kernel_regularizer=l2_reg_de, name="teacher_dense")(x)
        x = Reshape((3, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(filters[1], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(filters[0], (3, 3), strides=1, padding="same", kernel_regularizer=l2_reg_de, name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name=f"{self.name}_outputs",
        )(x)
        model = Model(inputs, outputs, name=self.name)
        if not self.compile: return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=[MeanSquaredError()])
        return model

class CicadaV1:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(inputs)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v1")

class CicadaV1scn:
    def __init__(self, input_shape: tuple, dense = [8], dropout = [0.1], search=False, compile=False, name: str = "cv1_scn"):
        self.input_shape = input_shape
        self.name = name
        self.dense = dense[0]
        self.dropout = dropout[0]
        self.search = search
        self.compile = compile

    def get_model(self, hp):
        if not self.search:
            dense = self.dense
            dropout = self.dropout
        else:
            dense = hp.Int("dense1", min_value=2, max_value=8., step=1)
            dropout = hp.Float("dropout", min_value=1e-10, max_value=1., step=10, sampling="log")
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = QDenseBatchnorm(
            dense,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(inputs)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(dropout)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        model = Model(inputs, outputs, name=self.name)
        if not self.compile: return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=[MeanAbsoluteError()])
        return model

class CicadaV2:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = Reshape((18, 14, 1), name="reshape")(inputs)
        x = QConv2D(
            4,
            (2, 2),
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            name="conv",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu0")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(1 / 9)(x)
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v2")

class CicadaV2scn:
    def __init__(self, input_shape: tuple, dense = [8], dropout = [0.1, 0.1], filters = [2], search=False, compile=False, name: str = "cv2_scn"):
        self.input_shape = input_shape
        self.name = name
        self.filters = filters[0]
        self.dropout1 = dropout[0]
        self.dropout2 = dropout[1]
        self.dense = dense[0]
        self.search = search
        self.compile = compile

    def get_model(self, hp):
        if not self.search:
            dense = self.dense
            dropout1 = self.dropout1
            dropout2 = self.dropout2
            filters = self.filters
        else:
            dense = [hp.Int("dense1", min_value=2, max_value=8, step=1)]
            dropout = [hp.Float("dropout1", min_value=1e-10, max_value=1., step=10, sampling="log"),
                       hp.Float("dropout2", min_value=1e-10, max_value=1., step=10, sampling="log")]
            filters = [hp.Int("conv", min_value=1, max_value=2, step=1)]

        inputs = Input(shape=self.input_shape, name="inputs_")
        x = Reshape((6, 14, 1), name="reshape")(inputs)
        x = QConv2D(
            filters,
            (2, 2),
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            name="conv",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu0")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(dropout1)(x)
        x = QDenseBatchnorm(
            dense,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(dropout2)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        model = Model(inputs, outputs, name=self.name)
        if not self.compile: return model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=[MeanAbsoluteError()])
        return model