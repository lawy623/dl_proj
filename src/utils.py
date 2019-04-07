import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import random
import os
from config import *
# from adv import *

def normalize(x):
    return x / tf.sqrt(tf.reduce_sum(x ** 2, axis=-1, keep_dims=True) + 1e-6)

def cossim(x, y, normalized=True):
    if normalized:
        return tf.reduce_sum(x * y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x ** 2) + 1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y ** 2) + 1e-6)
        return tf.reduce_sum(x * y) / x_norm / y_norm


def similarity(embedded, w, b, N=config.N, M=config.M, center=None):
    embedded_split = tf.reshape(embedded, shape=[N, M])
    print(embedded_split.shape)

    if center is None:
        center = tf.reduce_mean(embedded_split, axis=1)  # Shape = [N]
        print(center.shape)
        center_except = tf.reshape(tf.reduce_sum(embedded_split, axis=1, keep_dims=True) - embedded_split, shape=[N * M]) / (M - 1) # Shape = [N*M]
        print(center_except.shape)
        # shape = (N * M, N)
        S = tf.concat(
            [tf.concat([cossim(center_except[i * M:(i + 1) * M], embedded_split[j, :]) if i == j
                        else cossim(center[i:(i + 1)] * embedded_split[j, :]) for i in range(N)], axis=1) for j in range(N)], axis=0)
        print(S.shape)
    else:
        # center[i] * embedded_slit[j, :] is element-wise multiplication
        # therefore it needs to use reduce_sum to get vectors dot product
        S = tf.concat([
            tf.concat([cossim(center[i], embedded_split[j, :]) for i in range(N)], axis=1) for j in range(N)], axis=0)

    # shape = (N * M, N)
    S = tf.abs(w) * S + b
    return S


def loss_cal(S, name='softmax', N=config.N, M=config.M):
    # S_{j i, j}
    S_correct = tf.concat([S[i * M:(i + 1) * M, i:(i + 1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if name == "softmax":
        total = -tf.reduce_sum(S_correct - tf.log(tf.reduce_sum(tf.exp(S), axis=1, keep_dims=True) + 1e-6))
    elif name == "contrast":
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0 * S_sig[i * M:(i + 1) * M, j:(j + 1)] if i == j
                                      else S_sig[i * M:(i + 1) * M, j:(j + 1)] for j in range(N)], axis=1)
                           for i in range(N)], axis=0)
        total = tf.reduce_sum(1 - tf.sigmoid(S_correct) + tf.reduce_max(S_sig, axis=1, keep_dims=True))
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



if __name__ == "__main__":
    w= tf.constant([1], dtype= tf.float32)
    b= tf.constant([0], dtype= tf.float32)
    embedded = tf.constant([[0,1,0],[0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]], dtype= tf.float32)
    sim_matrix = similarity(embedded,w,b,3,2)
    loss1 = loss_cal(sim_matrix, type="softmax",N=3,M=2)
    loss2 = loss_cal(sim_matrix, type="contrast",N=3,M=2)
    with tf.Session() as sess:
        print(sess.run(sim_matrix))
        print(sess.run(loss1))
        print(sess.run(loss2))
