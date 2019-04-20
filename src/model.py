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
            self.buffer = Buffer() # Set a global buffer.
            self.batch = tf.placeholder(shape=[None, None, config.mels], dtype=tf.float32) # First dim is for n_frame  : config.N * config.M
            self.w = tf.get_variable('w', initializer=np.array([10], dtype=np.float32))
            self.b = tf.get_variable('b', initializer=np.array([-5], dtype=np.float32))
            self.lr = tf.placeholder(dtype=tf.float32)
            global_step = tf.Variable(0, name='global_step', trainable=False)

            self.embedded = self.build_model(self.batch) # Get the embedding representation.
            self.s_mat = similarity(self.embedded, w, b)
            if config.verbose:
                print('Embedded size: ', self.embedded.shape)
                print('Similarity matrix size: ', self.s_mat.shape)
            self.loss = loss_cal(self.s_mat, name=config.loss)

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
            self.batch = tf.placeholder(shape=[None, config.testN * config.testM * 2, config.mels], dtype=tf.float32)
            self.embedded = self.build_model(self.batch) # [2*testN*testM, nb_proj]
            # concatenate [enroll, verif]
            self.enroll_embed = normalize(tf.reduce_mean(
                tf.reshape(self.embedded[:config.testN * config.testM, :], shape=[config.testN, config.testM, -1]), axis=1))
            self.verif_embed = self.embedded[config.testN * config.testM:, :]

            self.s_mat = similarity(embedded=self.verif_embed, w=1.0, b=0.0, N=config.testN, M=config.testM, center=self.enroll_embed)

            if config.verbose:
                print('Embedded size: ', self.embedded.shape)
                print('Similarity matrix size: ', self.s_mat.shape)
        else:
            raise AssertionError("Please check the mode.")

        self.saver = tf.train.Saver()

    def build_model(self, batch):
        """
        Build the model. Choose one in the options.
        """
        return self.build_model_lstm(batch, use_attention = config.use_attention, use_mean = config.use_mean)
        #return self.build_model_bi_lstm(batch, use_attention = config.use_attention, use_mean = config.use_mean)

    def build_model_lstm(self, batch, use_attention = False, use_mean = True):
        """
        Deep learning model to extract the embedding and get the matrix.
        Model1: LSTM. 'attention' to choose whether to use attention.
        """
        print("Model Used: LSTM with projection...")
        with tf.variable_scope('lstm'):
            cells = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj)
                     for i in range(config.nb_layers)]
            lstm = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)
            if use_attention: # Use attention. final embedding is the weighted sum of all step's embedding.
                print("Use Attention in LSTM...")
                w_att = tf.Variable(tf.random.normal([config.nb_proj, config.att_size], stddev=0.1))
                b_att = tf.Variable(tf.random.normal([config.att_size], stddev=0.1))
                u_att = tf.Variable(tf.random.normal([config.att_size], stddev=0.1))
                v = tf.tanh(tf.tensordot(outputs, w_att, axes=1) + b_att)
                vu = tf.tensordot(v, u_att, axes=1)
                alphas = tf.nn.softmax(vu, axis=0, name='alphas')
                alphas = tf.expand_dims(tf.transpose(alphas, perm=[1,0]), -1)
                embedded = tf.reduce_sum(tf.transpose(outputs, perm = [1,0,2]) * alphas, 1)
            else:
                if use_mean:
                    embedded = tf.reduce_mean(outputs, axis=0)
                else:
                    embedded = outputs[-1]

            # shape = (N * M, nb_proj). Each e_ji is in (nb_proj,) dimension.
            embedded = normalize(embedded)
        return embedded

    def build_model_bi_lstm(self, batch, use_attention = False, use_mean = True):
        """
        Deep learning model to extract the embedding and get the matrix.
        Model2: Bi-LSTM. 'attention' to choose whether to use attention. 'use_mean' indicates whether to use the avg of embedding at all time steps.
        """
        print("Model Used: Bi-LSTM with projection...")
        with tf.variable_scope('bi-lstm'):
            cells_fw = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj) for i in range(config.nb_layers)]
            lstm_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
            cells_bw = [tf.contrib.rnn.LSTMCell(num_units=config.nb_hidden, num_proj=config.nb_proj) for i in range(config.nb_layers)]
            lstm_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw, cell_bw=lstm_bw, inputs=batch, dtype=tf.float32, time_major=True)
            outputs = tf.concat(outputs, 2)
            if use_attention: # Use attention. final embedding is the weighted sum of all step's embedding.
                print("Use Attention in LSTM...")
                w_att = tf.Variable(tf.random.normal([config.nb_proj*2, config.att_size*2], stddev=0.1))
                b_att = tf.Variable(tf.random.normal([config.att_size*2], stddev=0.1))
                u_att = tf.Variable(tf.random.normal([config.att_size*2], stddev=0.1))
                v = tf.tanh(tf.tensordot(outputs, w_att, axes=1) + b_att)
                vu = tf.tensordot(v, u_att, axes=1)
                alphas = tf.nn.softmax(vu, axis=0, name='alphas')
                alphas = tf.expand_dims(tf.transpose(alphas, perm=[1,0]), -1)
                embedded = tf.reduce_sum(tf.transpose(outputs, perm = [1,0,2]) * alphas, 1)
            else:
                if use_mean:
                    embedded = tf.reduce_mean(outputs, axis=0)
                else:
                    embedded = tf.math.add(outputs[0,:,:],outputs[-1,:,:]) / 2.0

            # shape = (N * M, nb_proj*2) due to combine embeddings. Each e_ji is in (nb_proj*2,) dimension.
            embedded = normalize(embedded)
        return embedded


    def train(self, sess):
        assert config.mode == 'train'
        sess.run(tf.global_variables_initializer())

        model_path = os.path.join(config.model_path, 'check_point')
        log_path = os.path.join(config.model_path, 'logs')

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        # Validation dataset
        enroll_valid_batch = get_test_batch(path = config.valid_path)
        verif_valid_batch = get_test_batch(path = config.valid_path, enroll = False)
        valid_batch = np.concatenate((enroll_valid_batch, verif_valid_batch), axis=1)

        writer = tf.summary.FileWriter(log_path, sess.graph)
        lr_factor = 1  ## Decay by half every n_decay iteration
        loss_acc = 0   ## accumulate loss in a number of iteration

        best_valid_EER = 1.0
        best_count = 0
        for i in range(int(config.nb_iters)):
            batch, _ = self.buffer.random_batch()
            _, loss_cur, summary = sess.run([self.train_op, self.loss, self.merged],
                                            feed_dict={self.batch: batch,
                                                       self.lr: config.lr * lr_factor})
            loss_acc += loss_cur # / (config.N * config.M) * 10 ## Norm the loss by dimension.

            if i % 10 == 0: # write to tensorboard
                writer.add_summary(summary, i)

            if (i + 1) % config.show_loss == 0: # print acc loss
                s = sess.run( (tf.reshape(self.s_mat, [config.N, config.M, -1]) - self.b) / tf.abs(self.w) )
                EER, THRES, EER_FAR, EER_FRR = cal_eer(s)
                if config.verbose: print('(iter : %d) loss: %.6f. | EER = %0.4f (thres:%0.2f, FAR:%0.4f, FRR:%0.4f).' % ((i + 1), loss_acc / (config.show_loss*config.N*config.M), EER,THRES,EER_FAR,EER_FRR) )
                loss_acc = 0

            if (i + 1) % config.decay == 0:  # decay lr by half.
                lr_factor /= 2
                if config.verbose: print('learning rate is decayed! current lr : ', config.lr * lr_factor)

            if (i + 1) % config.save_iters == 0: ## save model (!need to change log to value on validation)
                print("Validation...")
                EER = self.valid(sess, valid_batch)
                if EER < best_valid_EER:
                    self.saver.save(sess, os.path.join(model_path, 'model.ckpt'), global_step=best_count)
                    if config.verbose: print('Model {} is saved at {}! Best Valid EER now:{}'.format(best_count, model_path, EER))
                    best_count += 1
                    best_valid_EER = EER


    def valid(self, sess, valid_batch):
        assert config.mode == 'train'

        embedded = sess.run(self.embedded, feed_dict={self.batch: valid_batch})
        enroll_embed = normalize(tf.reduce_mean(
                tf.reshape(embedded[:config.testN * config.testM, :], shape=[config.testN, config.testM, -1]), axis=1))
        verif_embed = embedded[config.testN * config.testM:, :]

        s_mat = similarity(embedded=verif_embed, w=1.0, b=0.0, N=config.testN, M=config.testM, center=enroll_embed)

        s = sess.run(tf.reshape(s_mat, [config.testN, config.testM, -1]))
        loss = sess.run(loss_cal(s_mat, name=config.loss, N=config.testN, M=config.testM))
        EER, THRES, EER_FAR, EER_FRR = cal_eer(s)

        print("Validation:  EER = %0.4f (thres:%0.2f, FAR:%0.4f, FRR:%0.4f)."%(EER,THRES,EER_FAR,EER_FRR))
        print('Valid Loss: %.6f.' % ( loss/(config.testN*config.testM) ))
        return EER

    def test(self, sess, path):
        assert config.mode == 'test'

        print("Restoring model from: ", path)
        self.saver.restore(sess, path) # restore the model

        enroll_batch = get_test_batch()
        verif_batch = get_test_batch(enroll = False) ## Getting same N persons with different M utter.(TI-SV)
        test_batch = np.concatenate((enroll_batch, verif_batch), axis=1)

        s = sess.run(self.s_mat, feed_dict={self.batch: test_batch})
        loss = sess.run(loss_cal(s, name=config.loss, N=config.testN, M=config.testM))
        s = s.reshape([config.testN, config.testM, -1])

        EER, THRES, EER_FAR, EER_FRR = cal_eer(s)

        print("\nTesting:   EER = %0.4f (thres:%0.2f, FAR:%0.4f, FRR:%0.4f)"%(EER,THRES,EER_FAR,EER_FRR))
        print('Test Loss: %.6f. ' % ( loss/(config.testN*config.testM) ))

def cal_ff(s, thres):
        """
        Cal FAR and FRR
        """
        s_thres = s > thres

        far = sum([np.sum(s_thres[i]) - np.sum(s_thres[i, :, i]) for i in range(config.testN)]) / \
              (config.testN - 1) / config.testM / config.testN
        frr = sum([config.testM - np.sum(s_thres[i][:, i]) for i in range(config.testN)]) / config.testM / config.testN
        return far, frr

def cal_eer(s):
        """
        Calculate EER. Iterate all thes to get the one s.t. FAR==FRR.
        """
        diff = math.inf
        EER = 0; THRES = 0; EER_FAR=0; EER_FRR=0

        for thres in [0.01 * i + 0.5 for i in range(50)]:
            FAR, FRR = cal_ff(s, thres)

            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                THRES = thres
                EER = (FAR + FRR) / 2.0
                EER_FAR = FAR
                EER_FRR = FRR

        return EER, THRES, EER_FAR, EER_FRR
