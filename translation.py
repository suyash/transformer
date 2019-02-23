from absl import app, flags
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Softmax
from keras.models import Model

from data.translation import datasets, load_vocab, PAD_ID
from transformer import Transformer, label_smoothing

app.flags.DEFINE_string("model_dir", "models/sentiment",
                        "directory to save checkpoints and exported models")
app.flags.DEFINE_string("dataset", "wmt14", "translation dataset to use")
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
    source_input = Input((seq_len - 1, ))
    target_input = Input((seq_len - 1, ))
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
                                       target_id2word) = load_vocab(dataset)
    source_vocab_size, target_vocab_size = len(source_id2word), len(
        target_id2word)

    train_data, test_data = datasets(
        dataset,
        source_word2id,
        target_word2id,
        seq_len,
        test_files=["newstest2013"])

    train_data = train_data.map(
        lambda s, t: ((s[1:], t[:-1]),
                      label_smoothing(
                          tf.one_hot(t[1:], depth=target_vocab_size),
                          label_smoothing_epsilon, target_vocab_size)))
    test_data = test_data.map(
        lambda s, t: ((s[1:], t[:-1]),
                      label_smoothing(
                          tf.one_hot(t[1:], depth=target_vocab_size),
                          label_smoothing_epsilon, target_vocab_size)))

    train_data = train_data.shuffle(100).batch(batch_size).repeat()

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

    model.fit(
        train_data,
        steps_per_epoch=max_steps // 100,
        epochs=100,
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
    FLAGS = flags.FLAGS
    run(
        FLAGS.model_dir,
        FLAGS.dataset,
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
