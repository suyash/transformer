"""
imdb movie review sentiment prediction using the encoder
"""

from absl import app, flags

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Input, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformer import Encoder, Embedding, create_padding_mask

app.flags.DEFINE_string(
    "model_dir", "models/sentiment",
    "directory to save checkpoints and exported models")
app.flags.DEFINE_integer("vocab_size", 1024, "vocabulary size")
app.flags.DEFINE_integer("pad_id", 0, "pad id")
app.flags.DEFINE_integer("N", 1, "number of layers in the encoder")
app.flags.DEFINE_integer("seq_len", 256, "sequence length")
app.flags.DEFINE_integer("d_model", 128, "encoder model size")
app.flags.DEFINE_integer("d_ff", 512, "feedforward model size")
app.flags.DEFINE_integer("num_heads", 4, "number of attention heads")
app.flags.DEFINE_float("dropout", 0.1, "dropout")
app.flags.DEFINE_integer("batch_size", 64, "batch size")
app.flags.DEFINE_integer("epochs", 10, "number of training epochs")

FLAGS = flags.FLAGS


def create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h, dropout):
    inp = Input((seq_len, ))
    net = Embedding(vocab_size, d_model, pad_id)(inp)
    mask = Lambda(lambda t: create_padding_mask(t, pad_id))(inp)
    net = Encoder(
        N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)([net, mask])
    net = Flatten()(net)
    net = Dense(1, activation="sigmoid")(net)

    model = Model(inp, net)

    # NOTE: keras optimizers cannot be saved with optimizer state
    # need to use an optimizer from `tf.train`
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss="binary_crossentropy",
        metrics=["acc"])

    return model


def run(seq_len, vocab_size, pad_id, N, d_model, d_ff, h, dropout, model_dir,
        batch_size, epochs):
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    x_train = pad_sequences(
        x_train,
        maxlen=FLAGS.seq_len,
        padding="post",
        truncating="post",
        value=FLAGS.pad_id)
    x_test = pad_sequences(
        x_test,
        maxlen=FLAGS.seq_len,
        padding="post",
        truncating="post",
        value=FLAGS.pad_id)

    x_train[x_train >= FLAGS.vocab_size] = FLAGS.pad_id
    x_test[x_test >= FLAGS.vocab_size] = FLAGS.pad_id

    model = create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h,
                         dropout)
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            TensorBoard(
                log_dir=model_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=True)
        ])

    model.save_weights("%s/weights/model_weights" % model_dir)
    # tf.contrib.saved_model.save_keras_model(model, "%s/saved_model" % model_dir, serving_only=True)


def main(_):
    run(FLAGS.seq_len, FLAGS.vocab_size, FLAGS.pad_id, FLAGS.N, FLAGS.d_model,
        FLAGS.d_ff, FLAGS.num_heads, FLAGS.dropout, FLAGS.model_dir,
        FLAGS.batch_size, FLAGS.epochs)


if __name__ == "__main__":
    app.run(main)
