"""
- the usual way to do distributed keras is to convert it to an estimator using `model_to_estimator`
  but that does not work for subclassed models because of the same reasons model saving is not working

- keras has a [multi_gpu_model](https://keras.io/utils/#multi_gpu_model) wrapper for model, but that works
  for training when you have multiple GPUs on the same machine.

- Another approach offered is `DistributionStrategy` (https://www.tensorflow.org/guide/distribute_strategy),
  but that approach is also currently broken for subclassed models
  (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/python/keras_utils_test.py#L362-L391)

- There were 3 options,
  - Use Horovod from Uber, they provide some wrappers for keras' model.fit.
    Horovod is available on Azure (https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-tensorflow#horovod).

    This did not work out, first the implementation was broken for `tensorflow.keras` and required using the keras pip package.
    (https://github.com/Azure/MachineLearningNotebooks/issues/226).
    Did that in the `keras-team/keras` branch, but then the keras pip package's custom subclassed model had an issue
    with variable number of inner layers (https://github.com/keras-team/keras/issues/12334). So any kind of attention demo
    was impossible. Also, it looked like the weights were not being properly saved.

  - break out of keras and run a training loop (https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough)

    - Nice introduction to the concepts: https://youtu.be/la_M6bCV91M?t=239
    - DistributionStrategy: https://youtu.be/-h0cWBiQ8s8?t=56
    - This is written for Azure and based on https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training-with-deep-learning/distributed-tensorflow-with-parameter-server/tf_mnist_replica.py

    Tried this, there are a number of examples using tf.train.Supervisor, however that is deprecated. Tried to implement one using tf.train.MonitoredSession
    and was able to run a train loop, but the chief worker wouldn't exit. Seemed similar to https://github.com/tensorflow/tensorflow/issues/21745.

    Ran on TensorFlow 1.10, couldn't override that with 1.12 on azure (later figured out how to, but didn't try again).

  - break out of keras and run a training loop on Horovod TensorFlow

    Tried this also. Custom training samples use `tf.GradientTape`, which does not support recording operations for calculating gradients
    if it involves conditional operations (`tf.cond`, `tf.while`).

    Tried the normal way with using `model.input` as a placeholder for input, does train, but then cannot save model weights.
    The MonitoredSession errors out saying that the graph is finalized and the variables cannot be reinitialized.

- Now just doing Single Node Training on 4 K80 GPUs on Azure.

- Also, on the `non-subclassing` branch, after removing subclassing, created an estimator and did distributed training on horovod.
"""
from absl import app, flags
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Dense, Softmax
from tensorflow.keras.models import Model

from data.translation import datasets, load_vocab, PAD_ID
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
        epochs,
):
    (source_word2id, source_id2word), (target_word2id,
                                       target_id2word) = load_vocab(
                                           dataset, data_dir)
    source_vocab_size, target_vocab_size = len(source_id2word), len(
        target_id2word)

    train_data, test_data = datasets(
        dataset,
        data_dir,
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
        steps_per_epoch=max_steps // epochs,
        epochs=epochs,
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
        FLAGS.epochs,
    )


if __name__ == "__main__":
    app.run(main)
