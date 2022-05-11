# import the necessary packages
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from siamese_net import config
from siamese_net import metrics
from siamese_net import utils
from siamese_net.siamese_network import build_siamese_model

print("[INFO] loading CASIA-WebFace dataset...")

if not config.SAVED_DATA:
    dataset_dir = os.path.normcase(config.DATASET_DIR_PATH)
    dataset_label_file = os.path.normcase(config.DATASET_LABELS_PATH)
    with open(dataset_label_file, "r") as f:
        lines = f.readlines()
    size = len(lines)
    # random.shuffle(lines)
    print(f"[INFO] Loading Size of the data: {size}")

    images_data = np.zeros((size, *config.CL_IMG_SHAPE), dtype='uint8')
    labels = np.zeros(size, dtype='int32')
    count = [0] * 20000
    ten_percent_data = size * 0.1
    data_percent = ten_percent_data
    load_percent = 10
    data_load_time = time.time()
    for k in range(size):
        if k == data_percent:
            print(f"[INFO] Loaded {load_percent}% data")
            data_percent += ten_percent_data
            load_percent += 10
        line = lines[k]
        img_path, label = line.split(' ')
        index = int(label)
        img_path = os.path.normcase(img_path)
        image_data = cv2.imread(os.path.join(dataset_dir, img_path)).astype('uint8')
        image_data = cv2.resize(image_data, config.CL_IMG_SHAPE[::-1][1:])
        images_data[k] = image_data
        labels[k] = index
        count[index] = count[index] + 1
    np.savez(config.SAVED_DATASET_NAME, images_data=images_data, labels=labels)
    print(f"[INFO] Dataset loading took {time.time() - data_load_time} seconds")
else:
    data_load_time = time.time()
    images_labels_data = np.load(config.SAVED_DATASET_PATH)
    images_data = images_labels_data["images_data"]
    labels = images_labels_data["labels"]
    size = images_data.shape[0]
    print(f"[INFO] Dataset loading took {time.time() - data_load_time} seconds")

# add a channel dimension to the images
test_data_idx = int(size * 0.8)
valid_data_idx = int(test_data_idx * 0.8)
testX = images_data[test_data_idx:]
validX = images_data[valid_data_idx:test_data_idx]
trainX = images_data[:valid_data_idx]

testY = labels[test_data_idx:]
validY = labels[valid_data_idx:test_data_idx]
trainY = labels[:valid_data_idx]

trainX = trainX / 255.0
validX = validX / 255.0
testX = testX / 255.0


# prepare the positive and negative pairs
# print("[INFO] preparing positive and negative pairs...")
def data_generator(X, Y, batch_size=256):
    while True:
        x, y = utils.make_pairs(X, Y, batch_size)
        yield x, y


# configure the siamese network
print("[INFO] building siamese network...")
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    imgA = Input(shape=config.CL_IMG_SHAPE, name="input_imgA")
    imgB = Input(shape=config.CL_IMG_SHAPE, name="input_imgB")
    featureExtractor = build_siamese_model(config.CL_IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)
    # finally, construct the siamese network
    distance = Lambda(utils.euclidean_distance, name="Euclidean")([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance)
    model.summary()

# compile the model
print("[INFO] compiling model...")
model.compile(loss=metrics.contrastive_loss,
              optimizer=tf.keras.optimizers.Adam(0.001),
              # run_eagerly=True,
              )

# train the model
print("[INFO] training model...")

batch_size = config.TL_BATCH_SIZE
epochs = config.TL_EPOCHS
steps_per_epoch = int(trainX.shape[0] / batch_size)

train_batch_size = batch_size
valid_batch_size = int(batch_size / 8)
train_steps_per_epoch = int(trainX.shape[0] / train_batch_size)
valid_steps_per_epoch = int(validX.shape[0] / valid_batch_size)

train_data_generator = data_generator(trainX, trainY, train_batch_size)
valid_data_generator = data_generator(validX, validY, valid_batch_size)


# # Make a dataset given the directory and strategy
# def makeDataset(generator_func, X, Y, batch_size, strategy=None):
#
#     ds = tf.data.Dataset.from_generator(generator_func, args=[X, Y, batch_size], output_types=(
#     tf.float64, tf.float64))  # Make a dataset from the generator. MAKE SURE TO SPECIFY THE DATA TYPE!!!
#
#     options = tf.data.Options()
#     options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
#     ds = ds.with_options(options)
#
#     # Optional: Make it a distributed dataset if you're using a strategy
#     if strategy is not None:
#         ds = strategy.experimental_distribute_dataset(ds)
#
#     return ds
#
#
# training_ds = makeDataset(data_generator, trainX, trainY, train_batch_size, None)
# validation_ds = makeDataset(data_generator, validX, validY, valid_batch_size, None)

history = model.fit(
    train_data_generator,
    validation_data=valid_data_generator,
    # batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    validation_steps=valid_steps_per_epoch,
    epochs=epochs,
)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.CL_MODEL_PATH)
# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.CL_PLOT_PATH)
