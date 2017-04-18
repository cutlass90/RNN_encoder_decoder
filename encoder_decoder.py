import os
import time
import itertools as it
import math

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import log_loss

import tools
from tools import EpochFinished


class EncoderDecoder(object):

    def __init__(self, embedding_size, n_hidden_RNN=256, do_train=True):

        self.embedding_size = embedding_size
        self.n_hidden_RNN = n_hidden_RNN
        self.create_graph()
        if do_train: self.create_optimizer_graph(self.cost)
        sub_d = len(os.listdir('summary'))
        self.train_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d)+'/train/')
        self.test_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d)+'/test/')
        self.merged = tf.summary.merge_all()
        
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(init_op)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(),
                                    max_to_keep = 1000)

    # --------------------------------------------------------------------------
    def __enter__(self):
        return self

    # --------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()
        if self.sess is not None:
            self.sess.close()

    # --------------------------------------------------------------------------
    def create_graph(self):

        print('Create graph')
        self.input_graph()

        self.Z = self.greate_encoder_graph(inputs=self.question1,
            seq_lengths=self.seq_lengths1, n_layers=2, embedding_size=self.embedding_size)

        self.recover = self.greate_decoder_graph(encoded_state=self.Z,
            inputs=self.question1, seq_lengths=self.seq_lengths1, n_layers=1)

        self.cost = self.create_cost_graph(recover=self.recover, targets=self.question1)

        print('Done!')

    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        self.question1 = tf.placeholder(tf.float32,
            shape=[None, None, self.embedding_size],
            name='question1')

        self.question2 = tf.placeholder(tf.float32,
            shape=[None, None, self.embedding_size],
            name='question1')

        self.seq_lengths1 = tf.placeholder(tf.int32, name='seq_lengths1')
        
        self.seq_lengths2 = tf.placeholder(tf.int32, name='seq_lengths2')

        self.targets = tf.placeholder(tf.float32, name='targets')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
    
    # --------------------------------------------------------------------------
    def greate_encoder_graph(self, inputs, seq_lengths, n_layers, embedding_size):
        print('\tgreate_encoder_graph')
        with tf.variable_scope('encoder'):
            RNN_state = self.RNN_encoder(inputs=inputs,
                seq_lengths=seq_lengths, n_layers=n_layers)
            encoded_state = tf.layers.dense(
                inputs=RNN_state,
                units=embedding_size,
                activation=tf.nn.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            return encoded_state

    # --------------------------------------------------------------------------
    def RNN_encoder(self, inputs, seq_lengths, n_layers):
        print('\t\tRNN_encoder')
        with tf.variable_scope('RNN_encoder'):
            cell = tf.contrib.rnn.GRUCell(self.n_hidden_RNN)

            fw_cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,
                output_keep_prob=self.keep_prob)

            bw_cell = tf.contrib.rnn.MultiRNNCell([cell]*n_layers)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,
                output_keep_prob=self.keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=inputs,
                sequence_length=seq_lengths,
                dtype=tf.float32)
            return states[0][-1] + states[1][-1] # tuple of fw and bw states with shape b x hRNN

    # --------------------------------------------------------------------------
    def greate_decoder_graph(self, encoded_state, inputs, seq_lengths, n_layers):
        print('\tgreate_decoder_graph')
        # first step is zero-vectors
        inputs = tf.concat([tf.zeros_like(inputs[:,0:1,:]), inputs], axis=1)[:,:-1,:]

        print(encoded_state, inputs, seq_lengths)

        with tf.variable_scope('decoder'):
            decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoded_state)
            dec_cell = tf.contrib.rnn.GRUCell(self.embedding_size)
            # dec_cell = tf.contrib.rnn.MultiRNNCell([dec_cell]*n_layers)
            # dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                # output_keep_prob=self.keep_prob)

            recover, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
                cell=dec_cell,
                decoder_fn=decoder_fn,
                inputs=inputs,
                sequence_length=seq_lengths)

            return recover

    # --------------------------------------------------------------------------
    def create_cost_graph(self, recover, targets):
        print('\tcreate_cost_graph')
        self.mse = tf.reduce_mean(tf.square(recover - targets))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])

        tf.summary.scalar('MSE', self.mse)
        tf.summary.scalar('L2 loss', self.L2_loss)
        
        return self.mse + self.L2_loss
    
    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('\tcreate_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            self.train = optimizer.minimize(cost)

    ############################################################################  
    def save_model(self, path = 'beat_detector_model', step = None):
        p = self.saver.save(self.sess, path, global_step = step)
        print("\tModel saved in file: %s" % p)

    ############################################################################
    def load_model(self, path):
        #path is path to file or path to directory
        #if path it is path to directory will be load latest model
        load_path = os.path.splitext(path)[0]\
        if os.path.isfile(path) else tf.train.latest_checkpoint(path)
        print('try to load {}'.format(load_path))
        self.saver.restore(self.sess, load_path)
        print("Model restored from file %s" % load_path)

    ############################################################################
    def train_(self, data_loader, keep_prob, weight_decay, learn_rate_start,
        learn_rate_end, batch_size, n_iter, save_model_every_n_iter=1000,
        path_to_model='classifier'):

        print('\t\t\t\t----==== Training ====----')
        try:
            self.load_model(os.path.dirname(path_to_model))
        except:
            print('Can not load model {0}, starting new train'.format(path_to_model))
            
        start_time = time.time()
        b = math.log(learn_rate_start/learn_rate_end, n_iter) 
        a = learn_rate_start*math.pow(1, b)
        for current_iter in tqdm(range(n_iter)):
            learn_rate = a/math.pow((current_iter+1), b)
            batch = data_loader.train.sample_batch(batch_size)
            feedDict = {self.question1 : batch['questions_1'],
                        self.seq_lengths1 : batch['seq_lengths_1'],
                        self.keep_prob : keep_prob,
                        self.weight_decay : weight_decay,
                        self.learn_rate : learn_rate}
            _, summary = self.sess.run([self.train, self.merged],
                feed_dict = feedDict)
            self.train_writer.add_summary(summary, current_iter)


            batch = data_loader.test.sample_batch(batch_size)
            feedDict = {self.question1 : batch['questions_1'],
                        self.seq_lengths1 : batch['seq_lengths_1'],
                        self.keep_prob : 1,
                        self.weight_decay : weight_decay}
            _, summary = self.sess.run([self.cost, self.merged],
                feed_dict = feedDict)
            self.test_writer.add_summary(summary, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path = path_to_model, step = current_iter+1)

        self.save_model(path = path_to_model, step = current_iter+1)
        print('\tTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))

    ############################################################################
    def eval_cost(self, data_loader, batch_size, path_to_model):
        print('\t\t\t\t----==== Evaluating cost ====----')
        self.load_model(os.path.dirname(path_to_model))
        data_loader.train.i = 0
        data_loader.test.i = 0
        

        def eval(mode):

            result = np.empty([0])
            while True:
                try:
                    if mode=='train':
                        batch = data_loader.train.next_batch(batch_size)
                    elif mode=='test':
                        batch = data_loader.test.next_batch(batch_size)
                    else:
                        raise ValueError('error of mode type')
                except EpochFinished:
                    break
                feedDict = {self.question1 : batch['questions_1'],
                            self.question2 : batch['questions_2'],
                            self.seq_lengths1 : batch['seq_lengths_1'],
                            self.seq_lengths2 : batch['seq_lengths_2'],
                            self.keep_prob : 1}

                res = self.sess.run(self.preds, feed_dict = feedDict)
                result = np.concatenate([result, res])

            if mode=='train':
                y_true = data_loader.train_data['is_duplicate']
            elif mode=='test':
                y_true = data_loader.test_data['is_duplicate']
            else:
                raise ValueError('error of mode type')

            return log_loss(y_true=y_true, y_pred=result)

        print('train loss=', eval(mode='train'))
        print('test loss=', eval(mode='test'))          


    #############################################################################################################
    def predict(self, batch_size, data_loader, path_to_save, path_to_model):
        predicting_time = time.time()
        print('\t\t\t\t----==== Predicting beats ====----')
        self.load_model(os.path.dirname(path_to_model))
        
        result = []
        data_loader.train.i = 0

        forward_pass_time = 0
        for current_iter in it.count():
            try:
                batch = data_loader.train.next_batch(batch_size)
            except EpochFinished:
                break
            feedDict = {self.question1 : batch['questions_1'],
                        self.seq_lengths1 : batch['seq_lengths_1'],
                        self.keep_prob : 1}

            start_time = time.time()
            res = self.sess.run(self.recover, feed_dict = feedDict)
            result += [[w for w in res[i,:batch['seq_lengths_1'][i],:]]\
                for i in range(res.shape[0])]
            forward_pass_time = forward_pass_time + (time.time() - start_time)
        data_loader.save_vectors_as_words(path_to_save, result)
            
        print('\tfile saved ', path_to_save)

        print('forward_pass_time = ', forward_pass_time)
        print('predicting_time = ', time.time() - predicting_time)

# testing #####################################################################################################################
"""
path_to_file = '../data/test/AAO3CXJKEG.npy'
data = np.load(path_to_file).item()
n_chunks=128
overlap = 700 #in samples
chunked_data = utils.chunking_data(data)
"""
