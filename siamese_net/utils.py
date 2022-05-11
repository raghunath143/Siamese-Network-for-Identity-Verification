# import the necessary packages
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    classes = np.unique(labels)
    numClasses = len(classes)
    classes_to_idx = {classes[i]: i for i in range(0, numClasses)}

    idx = [np.where(labels == classes[i])[0] for i in range(0, numClasses)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[classes_to_idx[label]])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return np.array(pairImages), np.array(pairLabels)


def make_pairs(images, labels, batch_size):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    classes = np.unique(labels)
    numClasses = len(classes)
    classes_to_idx = {classes[i]: i for i in range(0, numClasses)}

    idx = [np.where(labels == classes[i])[0] for i in range(0, numClasses)]
    # loop over all images
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        idxA = random.randint(0, images.shape[0] - 1)
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[classes_to_idx[label]])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    pairImages = np.array(pairImages)
    pairLabels = np.array(pairLabels)
    return [pairImages[:, 0], pairImages[:, 1]], pairLabels[:]


def make_triplets(images, labels, batch_size=256, img_shape=(784,)):
    x_anchors = np.zeros((batch_size, *img_shape))
    x_positives = np.zeros((batch_size, *img_shape))
    x_negatives = np.zeros((batch_size, *img_shape))

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, images.shape[0] - 1)
        x_anchor = images[random_index]
        y = labels[random_index]

        indices_for_pos = np.squeeze(np.where(labels == y), axis=0)
        indices_for_neg = np.squeeze(np.where(labels != y), axis=0)

        x_positive = images[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = images[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

    return [x_anchors, x_positives, x_negatives]


def get_image_data(img_path, image_resize_shape):
    image_data = cv2.imread(img_path).astype('uint8')
    if len(image_resize_shape) == 3:
        image_resize_shape = image_resize_shape[::-1][1:]
        # image_resize_shape = image_resize_shape[:-1]
    else:
        image_resize_shape = image_resize_shape[::-1]
    image_resized = cv2.resize(image_data, image_resize_shape)
    image_resized = image_resized / 255.0
    return image_resized


def make_triplets_images(image_paths, labels, batch_size=256, img_shape=(784,)):
    x_anchors = np.zeros((batch_size, *img_shape))
    x_positives = np.zeros((batch_size, *img_shape))
    x_negatives = np.zeros((batch_size, *img_shape))

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, image_paths.shape[0] - 1)
        y = labels[random_index]

        indices_for_pos = np.squeeze(np.where(labels == y), axis=0)
        indices_for_neg = np.squeeze(np.where(labels != y), axis=0)

        x_anchor_path = image_paths[random_index]
        x_positive_path = image_paths[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative_path = image_paths[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        x_anchors[i] = get_image_data(x_anchor_path, img_shape)
        x_positives[i] = get_image_data(x_positive_path, img_shape)
        x_negatives[i] = get_image_data(x_negative_path, img_shape)

    return [x_anchors, x_positives, x_negatives]


def euclidean_distance_triplet(embeddings):
    # unpack the vectors into separate lists
    (featsA, featsB, featsC) = tf.split(embeddings, 3, axis=1, name='split')
    # compute the sum of squared distances between the vectors
    distAB = euclidean_distance((featsA, featsB))
    distAC = euclidean_distance((featsA, featsC))
    distBC = euclidean_distance((featsB, featsC))
    # output = concatenate([featsAnchor, featsPositive, featsNegative], axis=1)
    distances = K.concatenate([distAB, distAC, distBC], axis=1)
    return distances


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    if "loss" in H.history:
        plt.plot(H.history["loss"], label="train_loss")
    if "val_loss" in H.history:
        plt.plot(H.history["val_loss"], label="val_loss")
    if "accuracy" in H.history:
        plt.plot(H.history["accuracy"], label="train_acc")
    if "val_accuracy" in H.history:
        plt.plot(H.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
