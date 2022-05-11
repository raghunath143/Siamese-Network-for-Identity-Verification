# import the necessary packages
import tensorflow as tf


@tf.function
def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    # y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    y = tf.cast(y, preds.dtype)
    squaredPreds = tf.square(preds)
    squaredMargin = tf.square(tf.maximum(margin - preds, 0))
    loss = tf.reduce_mean(y * squaredPreds + (1 - y) * squaredMargin)
    # print("contrastive_loss")
    # return the computed contrastive loss to the calling function
    return loss


@tf.function
def triplet_loss(y_true, y_pred, alpha=0.2):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    # y = tf.cast(y, preds.dtype)
    # calculate the triplet loss between the true labels and
    # the predicted labels
    y_true = tf.cast(y_true, y_pred.dtype)  # Not used in the loss function
    positive_dist, negative_dist, positive_negative_dist = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    loss = tf.maximum(positive_dist - negative_dist + alpha, 0.)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def triplet_loss_emb(y_true, y_pred, alpha=0.2, emb_size=64):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    # y = tf.cast(y, preds.dtype)
    # calculate the triplet loss between the true labels and
    # the predicted labels
    y_true = tf.cast(y_true, y_pred.dtype)  # Not used in the loss function

    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(positive_dist - negative_dist + alpha, 0.)
    loss = tf.reduce_mean(loss)
    return loss
