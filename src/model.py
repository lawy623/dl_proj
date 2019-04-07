import tensorflow as tf
import numpy as np
import os
import math
from utils import *
from config import *
from adv import *
from batch import *

class Model:
    def __init__(self):
        """
        Basic Model Structure.
        """
        if config.mode == 'train':
            self.batch = tf.placeholder(shape=[None, config.N * config.M, config.mels], dtype=tf.float32) # First dim is for n_frame
            w = tf.get_variable('w', initializer=np.array([10], dtype=np.float32))
            b = tf.get_variable('b', initializer=np.array([-5], dtype=np.float32))
            self.lr = tf.placeholder(dtype=tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            embedded = self.build_model(self.batch) # Get the embedding representation.
            s_mat = similarity(embedded, w, b)
            if config.verbose:
                print('Embedded size: ', embedded.shape)
                print('Similarity matrix size: ', s_mat.shape)
            self.loss = loss_cal(s_mat, name=config.loss)

            # optimization
            trainable_vars = tf.trainable_variables()
            optimizer = optim(self.lr)
            grads, params = zip(*optimizer.compute_gradients(self.loss))
            grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)
            # 0.01 gradient scale for w and b, 0.5 gradient scale for projection nodes
            grads_rescale = [0.01 * g for g in grads_clip[:2]]
            for g, p in zip(grads_clip[2:], params[2:]):
                if 'projection' in p.name:
                    grads_rescale.append(0.5 * g)
                else:
                    grads_rescale.append(g)
            self.train_op = optimizer.apply_gradients(zip(grads_rescale, params), global_step=global_step)


            variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
            if config.verbose: print('Total variables:', variable_count)

            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

        elif config.mode == 'test':
            self.batch = tf.placeholder(shape=[None, config.N * config.M, config.mels], dtype=tf.float32)
            embedded = self.build_model(self.batch) # [2NM, nb_proj]
            # concatenate [enroll, verif]
            enroll_embed = normalize(tf.reduce_mean(
                tf.reshape(embedded[:config.N * config.M, :], shape=[config.N, config.M, -1]), axis=1))
            verif_embed = embedded[config.N * config.M:, :]

            self.s_mat = similarity(embedded=verif_embed, w=1.0, b=0.0, center=enroll_embed) # Shape??

            if config.verbose:
                print('Embedded size: ', embedded.shape) # [NM, P]
                print('Similarity matrix size: ', self.s_mat.shape)
        else:
            raise AssertionError("Please check the mode.")

        self.saver = tf.train.Saver()

    def build_model(self, batch):
        """
        Deep learning model to extract the embedding and get the matrix.
        Model1: LSTM
        """
        with tf.variable_scope('lstm'):
            cells = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj)
                     for i in range(config.nb_layers)]
            lstm = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)
            embedded = outputs[-1]

            # shape = (N * M, nb_proj). Each e_ji is in (nb_proj,) dimension.
            embedded = normalize(embedded)
        return embedded


    def train(self, sess):
        assert config.mode == 'train'
        sess.run(tf.global_variables_initializer())

        model_path = os.path.join(config.model_path, 'check_point')
        log_path = os.path.join(config.model_path, 'logs')

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        writer = tf.summary.FileWriter(log_path, sess.graph)
        lr_factor = 1  ## Decay by half every xx iteration
        loss_acc = 0   ## accumulate loss in every xx iteration
        for i in range(int(config.nb_iters)):
            batch, _ = random_batch()
            _, loss_cur, summary = sess.run([self.train_op, self.loss, self.merged],
                                            feed_dict={self.batch: batch,
                                                       self.lr: config.lr * lr_factor})
            loss_acc += loss_cur / (config.N * config.M) * 10 ## Norm the loss by dimension.

            if i % 10 == 0: # write to tensorboard
                writer.add_summary(summary, i)

            if (i + 1) % config.show_loss == 0: # print acc loss
                if config.verbose: print('(iter : %d) loss: %.4f' % ((i + 1), loss_acc / config.show_loss))
                loss_acc = 0

            if (i + 1) % config.decay == 0:  # decay lr by half.
                lr_factor /= 2
                if config.verbose: print('learning rate is decayed! current lr : ', config.lr * lr_factor)

            if (i + 1) % config.save_iters == 0: ## save model (!need to change log to value on validation)
                self.saver.save(sess, os.path.join(model_path, 'model.ckpt'), global_step=i // config.save_iters)
                if config.verbose: print('model is saved!')

    def test(self, sess, path):
        assert config.mode == 'test'

        self.saver.restore(sess, path) # restore the model

        enroll_batch = get_test_batch()
        verif_batch = get_test_batch(utter_start = config.M)
        test_batch = np.concatenate(enroll_batch, verif_batch, axis=1)

        s = sess.run(self.s_mat, feed_dict={self.batch: test_batch})
        s = s.reshape([config.N, config.M, -1])

        EER, THRES, EER_FAR, EER_FRR = cal_eer(s)

        print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,THRES,EER_FAR,EER_FRR))

    def cal_ff(s, thres):
        """
        Cal FAR and FRR
        """
        s_thres = s > thres

        far = sum([np.sum(s_thres[i]) - np.sum(s_thres[i, :, i]) for i in range(config.N)]) / \
              (config.N - 1) / config.M / config.N
        frr = sum([config.M - np.sum(s_thres[i][:, i]) for i in range(config.N)]) / config.M / config.N
        return far, frr

    def cal_eer(s):
        """
        Calculate EER.
        """
        diff = math.inf
        EER = 0; THRES = 0; EER_FAR=0; EER_FRR=0

        for thres in [0.01 * i + 0.5 for i in range(50)]:
            FAR, FRR = cal_ff(s, thres)

            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                THRES = thres
                EER = (FAR + FRR) / 2.0
                EER_FAR = FAT
                EER_FRR = FRR

        return EER, THRES, EER_FAR, EER_FRR
