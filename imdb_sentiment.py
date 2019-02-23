"""
imdb movie review sentiment prediction using the encoder
"""

from absl import app, flags
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.layers import Add, Dense, Dropout, Flatten, Input, Lambda
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from transformer import Encoder, Embedding, create_padding_mask, PositionalEncoding

app.flags.DEFINE_string("model_dir", "models/sentiment",
                        "directory to save checkpoints and exported models")
app.flags.DEFINE_integer("vocab_size", 2048, "vocabulary size")
app.flags.DEFINE_integer("pad_id", 0, "pad id")
app.flags.DEFINE_integer("N", 2, "number of layers in the encoder")
app.flags.DEFINE_integer("seq_len", 256, "sequence length")
app.flags.DEFINE_integer("d_model", 128, "encoder model size")
app.flags.DEFINE_integer("d_ff", 512, "feedforward model size")
app.flags.DEFINE_integer("num_heads", 8, "number of attention heads")
app.flags.DEFINE_float("dropout", 0.5, "dropout")
app.flags.DEFINE_integer("batch_size", 250, "batch size")
app.flags.DEFINE_integer("epochs", 50, "number of training epochs")


def create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h, dropout):
    inp = Input((seq_len, ))
    embedding = Embedding(vocab_size, d_model, pad_id)(inp)
    encoding = PositionalEncoding(d_model)(inp)
    net = Add()([embedding, encoding])
    net = Dropout(dropout)(net)
    mask = Lambda(
        lambda t: create_padding_mask(t, pad_id), name="input_mask")(inp)
    net = Encoder(
        N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)([net, mask])
    net = Flatten()(net)
    net = Dense(2, activation="softmax")(net)

    model = Model(inp, net)

    # NOTE: keras optimizers cannot be saved with optimizer state
    # need to use an optimizer from `tf.train`
    # NOTE: this seems to be a 1.0 thing, in 2.0 all tf.train optimizers are
    # dropped and the keras versions are the only implementations
    # NOTE: this is not recommended for training, the paper authors describe
    # a variable learning rate schedule, that still needs to be implemented.
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-9)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model


def run(
        model_dir,
        seq_len,
        vocab_size,
        pad_id,
        N,
        d_model,
        d_ff,
        h,
        dropout,
        batch_size,
        epochs,
):
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    x_train = pad_sequences(
        x_train, maxlen=seq_len, padding="pre", truncating="pre", value=pad_id)
    x_test = pad_sequences(
        x_test, maxlen=seq_len, padding="pre", truncating="pre", value=pad_id)

    y_train = np.eye(2)[y_train.astype(np.int32)]
    y_test = np.eye(2)[y_test.astype(np.int32)]

    x_train[x_train >= vocab_size] = pad_id
    x_test[x_test >= vocab_size] = pad_id

    model = create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h,
                         dropout)
    model.summary()

    x_train, x_eval, y_train, y_eval = train_test_split(
        x_train, y_train, test_size=0.25, random_state=1)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_eval, y_eval),
        callbacks=[
            TensorBoard(
                log_dir=model_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=True)
        ])

    print("Test Results:", model.evaluate(x_test, y_test))

    model.save_weights("%s/weights/model_weights" % model_dir)
    # tf.contrib.saved_model.save_keras_model(model, "%s/saved_model" % model_dir, serving_only=True)


def main(_):
    FLAGS = flags.FLAGS
    run(FLAGS.model_dir, FLAGS.seq_len, FLAGS.vocab_size, FLAGS.pad_id,
        FLAGS.N, FLAGS.d_model, FLAGS.d_ff, FLAGS.num_heads, FLAGS.dropout,
        FLAGS.batch_size, FLAGS.epochs)


if __name__ == "__main__":
    app.run(main)
