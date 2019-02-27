import collections
import cPickle
import os
import re
import random
from io import open
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data_path, sample=True):
    with open (data_path, "r", encoding="utf-8") as f:
        if not sample:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]


def build_dict(sample=True):
    if os.path.exists("./data/sumdata/word_dict.pkl"):
        print("load dict")
        with open("./data/sumdata/word_dict.pkl", "rb") as f:
            word_dict = cPickle.load(f)
    else:
        print("build dict")
        train_article_list = get_text_list("data/sumdata/train/train.article.txt", sample)
        train_title_list = get_text_list("data/sumdata/train/train.title.txt", sample)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("./data/sumdata/word_dict.pkl", "wb") as f:
            cPickle.dump(word_dict, f)
    return word_dict


def build_dataset(word_dict, step='train', article_max_len=50, summary_max_len=15, sample=True):
    # word => #num of word, not cnt num
    if step == "train":
        print("building train dataset")
        print("preparing original data")
        article_list = get_text_list("data/sumdata/train/{}.article.txt".format(step), sample)
        title_list = get_text_list("data/sumdata/train/{}.title.txt".format(step), sample)

        print("process article data")
        train_x = [word_tokenize(d) for d in article_list]
        train_x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in train_x]
        train_x = [d[:article_max_len] for d in train_x]

        train_x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in train_x]
        train_x_len = map(lambda x: len([y for y in x if y != 0]), train_x)

        print("process title data")
        train_y = [word_tokenize(d) for d in title_list]
        train_y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in train_y]
        train_y = [d[:(summary_max_len - 1)] for d in train_y]

        train_decoder_input = [[word_dict["<s>"]] + d for d in train_y]
        train_decoder_input = [d + (summary_max_len - len(d)) * [word_dict["<padding>"]] for d in train_decoder_input]
        train_decoder_output = [d + [word_dict["</s>"]]for d in train_y]
        train_decoder_output = [d + (summary_max_len - len(d)) * [word_dict["<padding>"]] for d in train_decoder_output]
        train_decoder_len = map(lambda x: len([y for y in x if y != 0]), train_decoder_input)

        with open("./data/sumdata/word_dataset_train.pkl", "wb") as f:
            cPickle.dump((train_x, train_x_len, train_decoder_input, train_decoder_output, train_decoder_len), f)
    elif step == "test":
        print("build test dataset")
        article_list = get_text_list("data/sumdata/train/{}.article.txt".format(step), sample)

        train_x = [word_tokenize(d) for d in article_list]
        train_x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in train_x]
        train_x = [d[:article_max_len] for d in train_x]

        train_x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in train_x]
        train_x_len = map(lambda x: len([y for y in x if y != 0]), train_x)
        
        with open("./data/sumdata/word_dataset_test.pkl", "wb") as f:
            cPickle.dump((train_x, train_x_len), f)
    else:
        raise NotImplementedError


def build_init_embedding(embedding_size=300):
    print("Loading dict...")
    with open("./data/sumdata/word_dict.pkl", "rb") as f:
        word_dict = cPickle.load(f)
    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    print("Loading Glove vectors...")
    glove_file = "./data/sumdata/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)
    word_vec_arr = np.array(word_vec_list)

    with open("./data/sumdata/glove.pkl", "wb") as f:
        cPickle.dump(word_vec_arr, f)
    return word_vec_arr


def convert(source_dir, target_dir, num_shards=128, tfrecords_prefix='', step='train'):
    if not tf.gfile.Exists(source_dir):
        raise Exception('source dir {} does not exists'.format(source_dir))
    
    if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
        tfrecords_prefix += '-'

    target_dir = target_dir+"_"+step
    if tf.gfile.Exists(target_dir):
        tf.gfile.DeleteRecursively(target_dir)
    tf.gfile.MakeDirs(target_dir)

    if os.path.exists(os.path.join(source_dir, "word_dataset_{}.pkl".format(step))):
        with open(os.path.join(source_dir, "word_dataset_{}.pkl".format(step)), "rb") as f:
            print("load {} dataset".format(step))
            if step == 'train':
                x, x_len, decoder_input, decoder_output, decoder_len = cPickle.load(f)
            elif step == 'test':
                x, x_len = cPickle.load(f)
    else:
        raise Exception("do not find word_dataset_train.pkl")

    # np.random.shuffle(path_list)
    group = zip(x, x_len, decoder_input, decoder_output, decoder_len)
    random.shuffle(group)   #in-place func
    num_files = len(x)
    num_per_shard = num_files // num_shards

    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))

    # convert to tfrecords
    shard_idx = 0
    writer = None
    if step == 'train':
        for i, (i_x, i_x_len, i_decoder_input, i_decoder_output, i_decoder_len) in enumerate(group):
            if i % num_per_shard == 0 and shard_idx < num_shards:
                shard_idx += 1
                tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
                tfrecord_path = os.path.join(target_dir, tfrecord_fn)
                print("Writing {} ...".format(tfrecord_path))
                if shard_idx > 1:
                    writer.close()
                writer = tf.python_io.TFRecordWriter(tfrecord_path)

            example = tf.train.Example(features=tf.train.Features(feature={
                # "ind":  _int64_features([i]),
                "batch_x": _bytes_features([np.array(i_x, dtype=np.int32).tobytes()]),
                "batch_x_len": _int64_features([i_x_len]),
                "batch_decoder_input": _bytes_features([np.array(i_decoder_input, dtype=np.int32).tobytes()]),
                "batch_decoder_output":_bytes_features([np.array(i_decoder_output, dtype=np.int32).tobytes()]),
                "batch_decoder_len": _int64_features([i_decoder_len]),
            }))
            writer.write(example.SerializeToString())
    else:
        for i, (i_x, i_x_len) in enumerate(zip(x, x_len)):
            if i % num_per_shard == 0 and shard_idx < num_shards:
                shard_idx += 1
                tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
                tfrecord_path = os.path.join(target_dir, tfrecord_fn)
                print("Writing {} ...".format(tfrecord_path))
                if shard_idx > 1:
                    writer.close()
                writer = tf.python_io.TFRecordWriter(tfrecord_path)

            example = tf.train.Example(features=tf.train.Features(feature={
                "batch_x": _bytes_features([np.array(i_x, dtype=np.int32).tobytes()]),
                "batch_x_len": _int64_features([i_x_len]),
            }))
            writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    # word_dict = build_dict(sample=True)
    # build_init_embedding()
    # build_dataset(word_dict, sample=True, step='train')
    convert('./data/sumdata', './data/sumdata/tfrecords', num_shards=16, tfrecords_prefix='sumdata', step='train')

