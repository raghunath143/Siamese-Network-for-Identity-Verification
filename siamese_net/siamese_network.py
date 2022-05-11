# import the necessary packages
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model

SIMPLE_MODEL = False


def build_siamese_model(inputShape, embeddingDim=64):
    if SIMPLE_MODEL:
        model = build_siamese_model_simple(inputShape, embeddingDim)
    else:
        model = build_siamese_model_complex(inputShape, embeddingDim)
    model.summary()
    return model


def build_siamese_model_complex(inputShape, embeddingDim=64):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu", name="Conv1")(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool1")(x)
    x = Dropout(0.3, name="DropOut1")(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (2, 2), padding="same", activation="relu", name="Conv2")(x)
    x = MaxPooling2D(pool_size=2, name="MaxPool2")(x)
    x = Dropout(0.3, name="DropOut2")(x)
    # third set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(256, (2, 2), padding="same", activation="relu", name="Conv3")(x)
    x = MaxPooling2D(pool_size=2, name="MaxPool3")(x)
    x = Dropout(0.3, name="DropOut3")(x)
    # fourth set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(1024, (2, 2), padding="same", activation="relu", name="Conv4")(x)
    x = MaxPooling2D(pool_size=2, name="MaxPool4")(x)
    x = Dropout(0.3, name="DropOut4")(x)
    # prepare the final outputs
    pooled_output = GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
    outputs = Dense(embeddingDim, name="SiameseEmbedding")(pooled_output)
    # build the model
    model = Model(inputs, outputs, name="Siamese", )
    # return the model to the calling function
    return model


def build_siamese_model_simple(inputShape, embeddingDim=64):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)
    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu", name="Conv1")(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name="MaxPool1")(x)
    x = Dropout(0.3, name="DropOut1")(x)
    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (2, 2), padding="same", activation="relu", name="Conv2")(x)
    x = MaxPooling2D(pool_size=2, name="MaxPool2")(x)
    x = Dropout(0.3, name="DropOut2")(x)
    # prepare the final outputs
    pooled_output = GlobalAveragePooling2D(name="GlobalAveragePooling2D")(x)
    outputs = Dense(embeddingDim, name="SiameseEmbedding")(pooled_output)
    # build the model
    model = Model(inputs, outputs, name="Siamese", )
    # return the model to the calling function
    return model
