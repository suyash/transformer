"""
imdb movie review sentiment prediction using the encoder
"""

import functools

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
app.flags.DEFINE_integer("max_steps", 1000, "maximum number of training steps")


def create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h, dropout):
    inp = Input((seq_len, ), name="input_text")
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
        max_steps,
):
    model = create_model(seq_len, vocab_size, pad_id, N, d_model, d_ff, h,
                         dropout)
    model.summary()

    tok = tokenizer(vocab_size, seq_len, data_dir=model_dir)

    train_input_fn_ = functools.partial(train_input_fn, tok, batch_size)
    test_input_fn_ = functools.partial(test_input_fn, tok, batch_size)

    config = tf.estimator.RunConfig(model_dir=model_dir)
    estimator = tf.keras.estimator.model_to_estimator(model, config=config)

    log_hook = tf.train.LoggingTensorHook(
        {"train_accuracy": "metrics/acc/Mean"}, every_n_iter=100)

    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name="loss",
        max_steps_without_decrease=500,
        min_steps=5000)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn_,
        max_steps=max_steps,
        hooks=[log_hook, early_stopping_hook])

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {"input_text": model.input})

    eval_spec = tf.estimator.EvalSpec(
        test_input_fn_,
        exporters=[
            tf.estimator.LatestExporter("model", serving_input_receiver_fn)
        ])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(_):
    FLAGS = flags.FLAGS
    run(FLAGS.model_dir, FLAGS.seq_len, FLAGS.vocab_size, FLAGS.pad_id,
        FLAGS.N, FLAGS.d_model, FLAGS.d_ff, FLAGS.num_heads, FLAGS.dropout,
        FLAGS.batch_size, FLAGS.max_steps)


if __name__ == "__main__":
    app.run(main)
