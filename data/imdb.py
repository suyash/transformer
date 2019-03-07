"""
tokenizing and loading imdb movie reviews dataset.
"""

import json
import os

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


def datasets(num_words, maxlen, data_dir=None):
    train_data, test_data = tfds.load("imdb_reviews", split=["train", "test"])
    tok = Tokenizer(num_words, maxlen, data_dir=data_dir)

    tok.fit(map(lambda x: x["text"], tfds.as_numpy(train_data)))

    train_text = tok.transform(
        map(lambda x: x["text"], tfds.as_numpy(train_data)))
    train_labels = map(lambda x: x["label"], tfds.as_numpy(train_data))

    test_text = tok.transform(
        map(lambda x: x["text"], tfds.as_numpy(test_data)))
    test_labels = map(lambda x: x["label"], tfds.as_numpy(test_data))

    print("done creating generators")

    # Do not use Dataset.from_generator if you intend to serialize the graph and
    # restore it in a different environment
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
    return tf.data.Dataset.from_tensor_slices(
        (list(train_text),
         list(train_labels))), tf.data.Dataset.from_tensor_slices(
             (list(test_text), list(test_labels)))
