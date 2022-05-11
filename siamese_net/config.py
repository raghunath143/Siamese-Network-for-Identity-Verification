# import the necessary packages
import os

# define the path to the base output directory
BASE_OUTPUT = "output"

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100
FULL_DATA = True
SAVED_DATA = True
DATASET_DIR_PATH = "dataset/CASIA-WebFace"
DATASET_LABELS_PATH = "dataset/train.txt"
# SAVED_DATASET_PATH = "dataset/casia_images_labels_array_28x28x3.npz"
SAVED_DATASET_PATH = "dataset/casia_images_labels_array_64x64x3.npz"
SAVED_DATASET_NAME = "dataset/casia_images_labels_array_64x64x3"

# specify the shape of the inputs for our network
TL_EPOCHS = 100
TL_BATCH_SIZE = 1024
TL_IMG_SHAPE = (64, 64, 3)
# TL_IMG_SHAPE = (28, 28, 3)

# use the base output path to derive the path to the serialized
# model along with training history plot
TL_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "triplet_siamese_model.h5"])
TL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "triplet_plot.png"])

# specify the shape of the inputs for our network
CL_EPOCHS = 100
CL_BATCH_SIZE = 1024
CL_IMG_SHAPE = (64, 64, 3)
# CL_IMG_SHAPE = (28, 28, 3)

# use the base output path to derive the path to the serialized
# model along with training history plot
CL_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_siamese_model.h5"])
CL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_plot.png"])

# specify the shape of the inputs for our network
BCEL_EPOCHS = 100
BCEL_BATCH_SIZE = 1024
BCEL_IMG_SHAPE = (64, 64, 3)
# BCEL_IMG_SHAPE = (28, 28, 3)
# use the base output path to derive the path to the serialized
# model along with training history plot
BCEL_MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "crossentropy_siamese_model.h5"])
BCEL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "crossentropy_plot.png"])
