import os

from absl import app, flags, logging
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input  # pylint: disable=import-error
import tensorflow_datasets as tfds

from .transformer import Encoder, PaddingMask


def main(_):
    data, info = tfds.load("imdb_reviews/subwords8k",
                           with_info=True,
                           as_supervised=True,
                           data_dir=flags.FLAGS.tfds_data_dir)

    train_data, test_data = data[tfds.Split.TRAIN], data[tfds.Split.TEST]

    train_data = train_data.filter(
        lambda x, y: tf.shape(x)[0] < flags.FLAGS.max_len)
    train_data = train_data \
        .padded_batch(flags.FLAGS.batch_size, train_data.output_shapes) \
        .shuffle(flags.FLAGS.shuffle_buffer_size) \
        .repeat()

    test_data = test_data.filter(
        lambda x, y: tf.shape(x)[0] < flags.FLAGS.max_len)
    test_data = test_data \
        .padded_batch(flags.FLAGS.batch_size, test_data.output_shapes)

    vocab_size = info.features["text"].encoder.vocab_size

    inp = Input((None, ), dtype="int32", name="inp")
    mask = PaddingMask()(inp)
    net, enc_enc_attention_weights = Encoder(
        num_layers=flags.FLAGS.num_layers,
        d_model=flags.FLAGS.d_model,
        num_heads=flags.FLAGS.num_heads,
        d_ff=flags.FLAGS.d_ff,
        vocab_size=vocab_size,
        dropout_rate=flags.FLAGS.dropout_rate)(inp, mask)
    net = GlobalAveragePooling1D()(net)
    net = Dense(1, activation="sigmoid")(net)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=flags.FLAGS.learning_rate)
    loss_object = tf.keras.losses.BinaryCrossentropy()

    if flags.FLAGS.use_custom_training_loop:
        model = Model(inputs=inp, outputs=[net, enc_enc_attention_weights])
        model.summary()

        train(train_data=train_data,
              validation_data=test_data,
              model=model,
              loss_object=loss_object,
              optimizer=optimizer,
              max_steps=flags.FLAGS.epochs * flags.FLAGS.steps_per_epoch,
              save_summary_steps=flags.FLAGS.steps_per_epoch,
              validation_steps=flags.FLAGS.validation_steps,
              job_dir=flags.FLAGS["job-dir"].value)
    else:
        model = Model(inputs=inp, outputs=net)
        model.summary()

        model.compile(optimizer=optimizer,
                      loss=loss_object,
                      metrics=[tf.keras.metrics.BinaryAccuracy()])

        model.fit(train_data,
                  epochs=flags.FLAGS.epochs,
                  steps_per_epoch=flags.FLAGS.steps_per_epoch,
                  validation_data=test_data,
                  validation_steps=flags.FLAGS.validation_steps,
                  callbacks=[
                      ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=5,
                                        min_lr=1e-6,
                                        verbose=1,
                                        cooldown=2),
                      TensorBoard(log_dir=flags.FLAGS["job-dir"].value),
                  ])

    model.save(os.path.join(flags.FLAGS["job-dir"].value, "model"))


@tf.function
def train_step(inp, tar, model, loss_object, optimizer, loss_mean, acc):
    with tf.GradientTape() as tape:
        out, _ = model(inp, training=True)
        loss = loss_object(y_true=tf.expand_dims(tar, 1), y_pred=out)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_mean(loss)
    acc(y_true=tar, y_pred=out)


def train(train_data, validation_data, model, loss_object, optimizer,
          max_steps, save_summary_steps, validation_steps, job_dir):
    loss_mean = tf.keras.metrics.Mean()
    acc = tf.keras.metrics.BinaryAccuracy()

    with tf.summary.create_file_writer(job_dir).as_default():  # pylint: disable=not-context-manager
        for step, (inputs, outputs) in enumerate(train_data):
            train_step(inputs,
                       outputs,
                       model=model,
                       loss_object=loss_object,
                       optimizer=optimizer,
                       loss_mean=loss_mean,
                       acc=acc)

            if step % save_summary_steps == 0:
                logging.info("Step: %d: Loss: %f, Accuracy: %f", step,
                             loss_mean.result(), acc.result())
                tf.summary.scalar("Train Loss", loss_mean.result(), step=step)
                tf.summary.scalar("Train Accuracy", acc.result(), step=step)

                loss_mean.reset_states()
                acc.reset_states()

                current_validation_step = 0
                for current_validation_step, (
                        x, y_true) in enumerate(validation_data):
                    y_pred, _ = model(x, training=False)
                    loss = loss_object(y_true=tf.expand_dims(y_true, 1),
                                       y_pred=y_pred)
                    loss_mean(loss)
                    acc(y_true, y_pred)

                    if current_validation_step >= validation_steps:
                        break

                logging.info(
                    "Step: %d, validation_loss: %f, validation accuracy: %f",
                    step, loss_mean.result(), acc.result())
                tf.summary.scalar("Validation Loss",
                                  loss_mean.result(),
                                  step=step)
                tf.summary.scalar("Validation Accuracy",
                                  acc.result(),
                                  step=step)
                loss_mean.reset_states()
                acc.reset_states()

            if step >= max_steps:
                break


if __name__ == "__main__":
    app.flags.DEFINE_integer("d_model", 128, "d_model")
    app.flags.DEFINE_integer("d_ff", 512, "d_ff")
    app.flags.DEFINE_integer("num_layers", 2, "num_layers")
    app.flags.DEFINE_integer("num_heads", 8, "num_heads")
    app.flags.DEFINE_float("dropout_rate", 0.1, "dropout_rate")
    app.flags.DEFINE_float("learning_rate", 1e-4, "learning_rate")
    app.flags.DEFINE_integer("epochs", 50, "epochs")
    app.flags.DEFINE_integer("steps_per_epoch", 250, "steps_per_epoch")
    app.flags.DEFINE_integer("max_len", 500, "max_len")
    app.flags.DEFINE_integer("batch_size", 4, "batch_size")
    app.flags.DEFINE_integer("shuffle_buffer_size", 500, "shuffle_buffer_size")
    app.flags.DEFINE_integer("validation_steps", 50, "validation_steps")
    app.flags.DEFINE_boolean("use_custom_training_loop", False,
                             "use_custom_training_loop")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string("job-dir", "runs/text_classification", "job-dir")
    app.run(main)
