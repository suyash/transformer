"""
tokenizing and loading imdb movie reviews dataset.
"""

import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class Tokenizer:
    """
    There is some ambiguity around the one shipping in tf.keras.utils,
    and it is a simple thing to implement

    0 => PAD
    1 => UNK

    numbering starts from 2 onwards
    also filters and num_words

    TODO: subword segmentation
    """

    def __init__(self,
                 num_words,
                 maxlen,
                 data_dir=None,
                 filter='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        self.num_words = num_words
        self.maxlen = maxlen
        self.data_dir = data_dir
        self.filter = {ord(c): None for c in filter}

        self.frequencies = {}
        self.index = {}

    def fit(self, g):
        for t in g:
            for word in t.split(b" "):
                word = word.decode("utf-8")
                word = word.lower()
                word = word.translate(self.filter)
                if len(word) == 0:
                    continue

                if word in self.frequencies:
                    self.frequencies[word] += 1
                else:
                    self.frequencies[word] = 1

        inverted = [(v, k) for k, v in self.frequencies.items()]
        inverted = sorted(inverted, key=lambda x: x[0])
        inverted = reversed(inverted)

        for i, (_, k) in enumerate(inverted):
            self.index[k] = i + 2

        if self.data_dir:
            d = os.path.join(self.data_dir, "data")

            try:
                os.makedirs(d)
            except:
                pass

            with open(os.path.join(d, "word_index.json"), "w") as f:
                json.dump(self.index, f)

    def transform(self, g):
        for t in g:
            cans = []

            for word in t.split(b" "):
                word = word.decode("utf-8")
                word = word.lower()
                word = word.translate(self.filter)
                if len(word) == 0:
                    continue

                if word in self.index and self.index[word] < self.num_words:
                    cans.append(self.index[word])
                else:
                    cans.append(1)  # UNK

            if len(cans) > self.maxlen:
                cans = cans[:self.maxlen]
            elif len(cans) < self.maxlen:
                cans = cans + ([0] * (self.maxlen - len(cans)))  # PAD

            yield cans

    def load_index(self, filepath):
        with open(filepath, "r") as f:
            self.index = json.load(f)


def tokenizer(num_words, maxlen, data_dir=None):
    train_data = tfds.load("imdb_reviews", split="train")
    tok = Tokenizer(num_words, maxlen, data_dir=data_dir)
    tok.fit(map(lambda x: x["text"], tfds.as_numpy(train_data)))
    return tok


def train_input_fn(tok, batch_size):
    train_data = tfds.load("imdb_reviews", split="train")
    train_data_np = list(tfds.as_numpy(train_data))
    train_text = tok.transform(map(lambda x: x["text"], train_data_np))
    train_labels = map(lambda x: np.eye(2)[x["label"]], train_data_np)
    train_labels = np.stack(list(train_labels), axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((list(train_text),
                                                  train_labels))
    dataset = dataset.batch(batch_size).repeat()
    return dataset


def test_input_fn(tok, batch_size):
    test_data = tfds.load("imdb_reviews", split="test")
    test_data_np = list(tfds.as_numpy(test_data))
    test_text = tok.transform(map(lambda x: x["text"], test_data_np))
    test_labels = map(lambda x: np.eye(2)[x["label"]], test_data_np)
    test_labels = np.stack(list(test_labels), axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((list(test_text),
                                                  test_labels))
    dataset = dataset.batch(batch_size)
    return dataset
