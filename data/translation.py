"""
https://github.com/lilianweng/transformer-tensorflow/blob/master/train.py
"""

import os
import sys
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf

PAD_ID = 0
UNKNOWN_ID = 1
START_ID = 2
END_ID = 3

config = {
    'iwslt15': {
        'source_lang':
        'en',
        'target_lang':
        'vi',
        'url':
        "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/",
        'files': [
            'train.en', 'train.vi', 'tst2012.en', 'tst2012.vi', 'tst2013.en',
            'tst2013.vi', 'vocab.en', 'vocab.vi'
        ],
        'train':
        'train',
        'test': ['tst2012', 'tst2013'],
        'vocab':
        'vocab',
    },
    'wmt14': {
        'source_lang':
        'en',
        'target_lang':
        'de',
        'url':
        "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/",
        'files': [
            'train.en', 'train.de', 'train.align', 'newstest2012.en',
            'newstest2012.de', 'newstest2013.en', 'newstest2013.de',
            'newstest2014.en', 'newstest2014.de', 'newstest2015.en',
            'newstest2015.de', 'vocab.50K.en', 'vocab.50K.de', 'dict.en-de'
        ],
        'train':
        'train',
        'test':
        ['newstest2012', 'newstest2013', 'newstest2014', 'newstest2015'],
        'vocab':
        'vocab.50K',
    },
    'wmt15': {
        'source_lang':
        'en',
        'target_lang':
        'cs',
        'url':
        "https://nlp.stanford.edu/projects/nmt/data/wmt15.en-cs/",
        'files': [
            'train.en', 'train.cs', 'newstest2013.en', 'newstest2013.cs',
            'newstest2014.en', 'newstest2014.cs', 'newstest2015.en',
            'newstest2015.cs', 'vocab.1K.en', 'vocab.1K.cs', 'vocab.10K.en',
            'vocab.10K.cs', 'vocab.20K.en', 'vocab.20K.cs', 'vocab.50K.en',
            'vocab.50K.cs'
        ],
        'train':
        'train',
        'test': ['newstest2013', 'newstest2014', 'newstest2015'],
        'vocab':
        'vocab.50K',
    }
}


def _maybe_download(dataset):
    for file in config[dataset]["files"]:
        _maybe_download_file(file, "data/%s" % dataset, config[dataset]["url"])


def _maybe_download_file(filename, data_dir, url_root):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' %
                (filename,
                 float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        download_url = "%s/%s" % (url_root, filename)
        filepath, _ = urlretrieve(download_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def load_vocab(dataset="wmt14"):
    _maybe_download(dataset)
    source_word2id, source_id2word = _load_vocab_file(
        "data/%s/%s.%s" % (dataset, config[dataset]["vocab"],
                           config[dataset]["source_lang"]))
    target_word2id, target_id2word = _load_vocab_file(
        "data/%s/%s.%s" % (dataset, config[dataset]["vocab"],
                           config[dataset]["source_lang"]))
    return ((source_word2id, source_id2word), (target_word2id, target_id2word))


def _load_vocab_file(filepath):
    words = list(map(lambda w: w.strip().lower(), open(filepath)))
    words.insert(0, "<pad>")
    words = words[:4] + list(set(words[4:]))
    word2id = {word: i for i, word in enumerate(words)}
    id2word = words
    return word2id, id2word


def _file_to_dataset(filepath, word2id, seq_len):
    data = tf.data.TextLineDataset(filepath)
    data = data.map(lambda line: tf.strings.strip(line))
    data = data.map(
        lambda line: tf.py_func(lambda s: s.lower(), [line], tf.string))
    data = data.map(lambda line: tf.string_split([line], ' ', skip_empty=True))

    # cannot do lookup tables here as keras doesn't automatically run `tables_initializer`
    # https://github.com/tensorflow/tensorflow/issues/20158
    # table = tf.contrib.lookup.index_table_from_tensor(mapping=target_id2word, default_value=UNKNOWN_ID)
    # data = data.map(lambda line: tf.SparseTensor(line.indices, table.lookup(line.values), line.dense_shape))

    map_fn = lambda w: tf.py_func(lambda x: np.int32(word2id.get(x.decode("ascii"), UNKNOWN_ID)), w, tf.int32)
    data = data.map(
        lambda line: tf.SparseTensor(line.indices, tf.map_fn(map_fn, [line.values], tf.int32), line.dense_shape))
    data = data.map(
        lambda line: tf.sparse.to_dense(line, default_value=UNKNOWN_ID)[0])
    data = data.map(
        lambda line: tf.concat([[START_ID], line, [END_ID]], axis=0))

    # NOTE: need the values to be padded beforehand for the filter use later, hence no `padded_batch`
    data = data.map(
        lambda line: tf.concat([line, tf.ones((tf.maximum(0, seq_len - tf.shape(line)[0])), dtype=tf.int32) * PAD_ID], axis=0))

    return data


def datasets(dataset, source_word2id, target_word2id, seq_len):
    source_file = "data/%s/%s.%s" % (dataset, config[dataset]["train"],
                                     config[dataset]["source_lang"])
    target_file = "data/%s/%s.%s" % (dataset, config[dataset]["train"],
                                     config[dataset]["target_lang"])

    source_dataset = _file_to_dataset(source_file, source_word2id, seq_len)
    target_dataset = _file_to_dataset(target_file, target_word2id, seq_len)

    dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    dataset = dataset.filter(
        lambda s, l: tf.logical_and(tf.equal(tf.size(s), seq_len), tf.equal(tf.size(l), seq_len)))
    return dataset


if __name__ == "__main__":
    ((source_word2id, source_id2word), (target_word2id,
                                        target_id2word)) = load_vocab("wmt14")
    print(len(source_word2id), len(target_word2id))
