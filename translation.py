"""
Trained this on a distributed cluster of 4 K80 GPUs on horovod, for 1_000 steps.
"""
import functools

from absl import app, flags
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Softmax
from tensorflow.keras.models import Model

from data.translation import train_input_fn, test_input_fn, load_vocab, PAD_ID
from transformer import Transformer, label_smoothing

app.flags.DEFINE_string("model_dir", "models/translation",
                        "directory to save checkpoints and exported models")
app.flags.DEFINE_string("dataset", "wmt14", "translation dataset to use")
app.flags.DEFINE_string("data_dir", "./data/wmt14", "location of the data")
app.flags.DEFINE_integer("seq_len", 10, "sequence length")
app.flags.DEFINE_integer("N", 6, "number of layers in the encoder and decoder")
app.flags.DEFINE_integer("d_model", 512, "encoder model size")
app.flags.DEFINE_integer("d_ff", 2048, "feedforward model size")
app.flags.DEFINE_integer("num_heads", 8, "number of attention heads")
app.flags.DEFINE_float("dropout", 0.1, "dropout")
app.flags.DEFINE_float("label_smoothing_epsilon", 0.1,
                       "label smoothing epsilon")
app.flags.DEFINE_integer("batch_size", 256, "batch size")
app.flags.DEFINE_integer("max_steps", 300_000, "number of training steps")
app.flags.DEFINE_integer("epochs", 100,
                         "number of epochs to divide the training steps into")


def create_model(
        source_vocab_size,
        target_vocab_size,
        seq_len,
        N,
        d_model,
        d_ff,
        h,
        dropout,
):
    source_input = Input((seq_len - 1, ), name="input_text")
    target_input = Input((seq_len - 1, ), name="target_text")
    net = Transformer(
        input_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        pad_id=PAD_ID,
        N_encoder=N,
        N_decoder=N,
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        dropout=dropout)([source_input, target_input])
    net = Softmax()(net)
    model = Model([source_input, target_input], net)

    # NOTE: keras optimizers cannot be saved with optimizer state
    # need to use an optimizer from `tf.train`
    # NOTE: this seems to be a 1.0 thing, in 2.0 all tf.train optimizers are
    # dropped and the keras versions are the only implementations
    # NOTE: this is not recommended for training, the paper authors describe
    # a variable learning rate schedule, that still needs to be implemented.
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-9)

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

    return model


def run(
        model_dir,
        dataset,
        data_dir,
        seq_len,
        N,
        d_model,
        d_ff,
        h,
        dropout,
        label_smoothing_epsilon,
        batch_size,
        max_steps,
):
    (source_word2id, source_id2word), (target_word2id,
                                       target_id2word) = load_vocab(
                                           dataset, data_dir)
    source_vocab_size, target_vocab_size = len(source_id2word), len(
        target_id2word)

    model = create_model(
        source_vocab_size,
        target_vocab_size,
        seq_len,
        N,
        d_model,
        d_ff,
        h,
        dropout,
    )

    model.summary()

    config = tf.estimator.RunConfig(model_dir=model_dir)
    estimator = tf.keras.estimator.model_to_estimator(model, config=config)

    train_input_fn_ = functools.partial(
        train_input_fn,
        dataset=dataset,
        data_dir=data_dir,
        source_word2id=source_word2id,
        target_word2id=target_word2id,
        seq_len=seq_len,
        target_vocab_size=target_vocab_size,
        label_smoothing_epsilon=label_smoothing_epsilon,
        batch_size=batch_size)

    train_spec = tf.estimator.TrainSpec(train_input_fn_, max_steps=max_steps)

    eval_input_fn_ = functools.partial(
        test_input_fn,
        dataset=dataset,
        data_dir=data_dir,
        source_word2id=source_word2id,
        target_word2id=target_word2id,
        seq_len=seq_len,
        target_vocab_size=target_vocab_size,
        label_smoothing_epsilon=label_smoothing_epsilon,
        batch_size=batch_size,
        test_files=["newstest2013"])

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        {
            "input_text": model.inputs[0],
            "target_text": model.inputs[1]
        })

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn_,
        exporters=[
            tf.estimator.LatestExporter("model", serving_input_receiver_fn)
        ])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    test_input_fn_ = functools.partial(
        test_input_fn,
        dataset=dataset,
        data_dir=data_dir,
        source_word2id=source_word2id,
        target_word2id=target_word2id,
        seq_len=seq_len,
        target_vocab_size=target_vocab_size,
        label_smoothing_epsilon=label_smoothing_epsilon,
        batch_size=batch_size)

    print("Test Results:", estimator.evaluate(test_input_fn_))


def main(_):
    FLAGS = flags.FLAGS
    run(
        FLAGS.model_dir,
        FLAGS.dataset,
        FLAGS.data_dir,
        FLAGS.seq_len,
        FLAGS.N,
        FLAGS.d_model,
        FLAGS.d_ff,
        FLAGS.num_heads,
        FLAGS.dropout,
        FLAGS.label_smoothing_epsilon,
        FLAGS.batch_size,
        FLAGS.max_steps,
    )


if __name__ == "__main__":
    app.run(main)
