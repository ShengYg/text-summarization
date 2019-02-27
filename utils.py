from nltk.tokenize import word_tokenize
import re
import os
import collections
import cPickle
import numpy as np
from io import open
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

def get_init_embedding():
    if os.path.exists("./data/sumdata/glove.pkl"):
        with open("./data/sumdata/glove.pkl", "rb") as f:
            return cPickle.load(f)
    else:
        raise NotImplementedError


class DB(object):
    def __init__(self, data_path=None, batch_size=64, num_epochs=300, shuffle_buffer=500 , map_parallel=1, step='train'):

        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._shuffle_buffer = shuffle_buffer
        self._map_parallel = map_parallel
        self._step = step
        self._data_path = os.path.join('data', 'sumdata', 'tfrecords_{}'.format(step)) if not data_path else data_path

    def parse_fn_train(self, serial_exmp):  
        feats = tf.parse_single_example(serial_exmp, features={
            # "ind": tf.FixedLenFeature([], tf.int64),
            'batch_x': tf.FixedLenFeature([], tf.string),
            'batch_x_len': tf.FixedLenFeature([], tf.int64),
            'batch_decoder_input': tf.FixedLenFeature([], tf.string),
            "batch_decoder_output": tf.FixedLenFeature([], tf.string),
            "batch_decoder_len": tf.FixedLenFeature([], tf.int64),
        })

        # ind =                   tf.cast(feats['ind'], tf.int32)
        batch_x =               tf.reshape(tf.decode_raw(feats['batch_x'], tf.int32), (1,))
        batch_x_len =           tf.cast(feats['batch_x_len'], tf.int32)
        batch_decoder_input =   tf.reshape(tf.decode_raw(feats['batch_decoder_input'], tf.int32), (1,))
        batch_decoder_output =  tf.reshape(tf.decode_raw(feats['batch_decoder_output'], tf.int32), (1,))
        batch_decoder_len =     tf.cast(feats['batch_decoder_len'], tf.int32)
        return batch_x, batch_x_len, batch_decoder_input, batch_decoder_output, batch_decoder_len

    def parse_fn_test(self, serial_exmp):  
        feats = tf.parse_single_example(serial_exmp, features={
            'batch_x': tf.FixedLenFeature([], tf.string),
            'batch_x_len': tf.FixedLenFeature([], tf.int64),
        })

        batch_x =       tf.reshape(tf.decode_raw(feats['batch_x'], tf.int32), (1,))
        batch_x_len =   tf.cast(feats['batch_x_len'], tf.int32)
        return batch_x, batch_x_len


    def input_pipeline_new(self, tfrecords_list, shuffle=False, prefetch=False):
        dataset = tf.data.TFRecordDataset(tfrecords_list)
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self._shuffle_buffer, count=self._num_epochs))
        if self._step == 'train':
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                    map_func=self.parse_fn_train, 
                    batch_size=self._batch_size, 
                    num_parallel_batches=self._map_parallel,
                    drop_remainder=True))
        elif self._step == 'test':
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                    map_func=self.parse_fn_test, 
                    batch_size=self._batch_size, 
                    num_parallel_batches=self._map_parallel,
                    drop_remainder=True))
        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        
        return dataset

    def get_dataset(self):
        tfrecords_list = sorted(os.listdir(self._data_path))
        tfrecords_list = [os.path.join(self._data_path, item) for item in tfrecords_list]
        train_dataset = self.input_pipeline_new(tfrecords_list, shuffle=True, prefetch=True)
        return train_dataset
