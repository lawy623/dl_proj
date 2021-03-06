import tensorflow as tf
import numpy as np
import random
import os
from config import *

def normalize(x):
    return x / tf.sqrt(tf.reduce_sum(x ** 2, axis=-1, keepdims=True) + 1e-6)

def cossim(x, y, normalized=True):
    if normalized:
        return tf.reduce_sum(x * y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x ** 2) + 1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y ** 2) + 1e-6)
        return tf.reduce_sum(x * y) / x_norm / y_norm


def similarity(embedded, w, b, N=config.N, M=config.M, center=None):
    embedded_split = tf.reshape(embedded, shape=[N, M, -1])

    if center is None:
        # normalize the center in order to do easy cosin calculation.
        center = normalize(tf.reduce_mean(embedded_split, axis=1)) #[N, P]
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keepdims=True) - embedded_split, shape=[N * M, -1]) / (M - 1)) #[NM, P]

        # shape = (N * M, N)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keep_dims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exist, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    # shape = (N * M, N)
    S = tf.abs(w) * S + b
    return S


def loss_cal(S, name='softmax', N=config.N, M=config.M):
    # S_{j i, j}
    S_correct = tf.concat([S[i * M:(i + 1) * M, i:(i + 1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if name == "softmax":
        total = -tf.reduce_sum(S_correct - tf.log(tf.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6))
    elif name == "contrast":
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0 * S_sig[i * M:(i + 1) * M, j:(j + 1)] if i == j
                                      else S_sig[i * M:(i + 1) * M, j:(j + 1)] for j in range(N)], axis=1)
                           for i in range(N)], axis=0)
        total = tf.reduce_sum(1 - tf.sigmoid(S_correct) + tf.reduce_max(S_sig, axis=1, keepdims=True))
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total


def optim(lr):
    assert config.optim[0] in ['sgd', 'rmsprop', 'adam']
    if config.optim[0] == 'sgd':
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim[0] == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr, **config.optim[1])
    else:
        return tf.train.AdamOptimizer(lr, **config.optim[1])
