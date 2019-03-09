"""
imdb movie review sentiment prediction using the encoder
"""

from absl import app, flags
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Add, Dense, Dropout, Flatten, Input, Lambda
from tensorflow.keras.models import Model

from data.imdb import tokenizer, train_input_fn, test_input_fn
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
app.flags.DEFINE_integer("early_stopping_patience", 5,
                         "early stopping patience")
app.flags.DEFINE_integer("max_steps", 10_000, "number of training steps")
app.flags.DEFINE_integer("epochs", 50, "number of training epochs to divide steps into")


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
        early_stopping_patience,
        max_steps,
        epochs,
):
    model = create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h,
                         dropout)
    model.summary()

    tok = tokenizer(vocab_size, seq_len, model_dir)

    train_data = train_input_fn(tok, batch_size)
    test_data = test_input_fn(tok, batch_size)

    model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=max_steps // epochs,
        validation_data=test_data,
        validation_steps=25000 // batch_size,
        callbacks=[
            TensorBoard(
                log_dir=model_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=True),
            EarlyStopping(
                min_delta=0.1, patience=early_stopping_patience, verbose=1),
            # NOTE: not working with tf.train optimizers
            # ReduceLROnPlateau(
            #     factor=0.2, patience=5, min_lr=0.00001, verbose=1),
        ])

    model.save_weights("%s/weights/model_weights" % model_dir)
    # tf.contrib.saved_model.save_keras_model(model, "%s/saved_model" % model_dir, serving_only=True)


def main(_):
    FLAGS = flags.FLAGS
    run(FLAGS.model_dir, FLAGS.seq_len, FLAGS.vocab_size, FLAGS.pad_id,
        FLAGS.N, FLAGS.d_model, FLAGS.d_ff, FLAGS.num_heads, FLAGS.dropout,
        FLAGS.batch_size, FLAGS.early_stopping_patience, FLAGS.max_steps, FLAGS.epochs)


if __name__ == "__main__":
    app.run(main)
