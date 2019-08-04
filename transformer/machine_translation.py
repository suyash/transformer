import os

from absl import app, flags, logging
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Input  # pylint: disable=import-error
import tensorflow_datasets as tfds

from .transformer import Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return dict([("d_model", self.d_model.numpy()),
                     ("warmup_steps", self.warmup_steps)])


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, mask_val=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_val = mask_val

    def call(self, y_true, y_pred):
        ans = tf.keras.backend.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True, axis=-1)
        mask = tf.reshape(tf.not_equal(y_true, self.mask_val), tf.shape(ans))
        return tf.boolean_mask(ans, mask)

    def get_config(self):
        config = super().get_config()
        config["mask_val"] = self.mask_val
        return config


def main(_):
    tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(
        flags.FLAGS.subwords_en_file)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file(
        flags.FLAGS.subwords_pt_file)

    data = tfds.load('ted_hrlr_translate/pt_to_en',
                     as_supervised=True,
                     data_dir=flags.FLAGS.tfds_data_dir)
    train_data, val_data = data["train"], data["validation"]

    train_data = preprocess_dataset(train_data, tokenizer_pt, tokenizer_en,
                                    flags.FLAGS.max_len)
    train_data = train_data.cache()
    train_data = train_data.shuffle(
        flags.FLAGS.shuffle_buffer_size).padded_batch(flags.FLAGS.batch_size,
                                                      padded_shapes=((-1, ),
                                                                     (-1, )))
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    val_data = preprocess_dataset(val_data, tokenizer_pt, tokenizer_en,
                                  flags.FLAGS.max_len)

    val_data = val_data.padded_batch(flags.FLAGS.batch_size,
                                     padded_shapes=((-1, ), (-1, )))

    src = Input((None, ), dtype="int32", name="src")
    tar = Input((None, ), dtype="int32", name="tar")

    logits, enc_enc_attention, dec_dec_attention, enc_dec_attention = Transformer(
        num_layers=flags.FLAGS.num_layers,
        d_model=flags.FLAGS.d_model,
        num_heads=flags.FLAGS.num_heads,
        d_ff=flags.FLAGS.d_ff,
        input_vocab_size=tokenizer_pt.vocab_size + 2,
        target_vocab_size=tokenizer_en.vocab_size + 2,
        dropout_rate=flags.FLAGS.dropout_rate)(src, tar)

    learning_rate = CustomSchedule(flags.FLAGS.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    loss = CustomLoss()

    if flags.FLAGS.use_custom_training_loop:
        model = tf.keras.Model(inputs=[src, tar],
                               outputs=[
                                   logits, enc_enc_attention,
                                   dec_dec_attention, enc_dec_attention
                               ])
        model.summary()

        train(
            train_data=train_data,
            val_data=val_data,
            model=model,
            validation_steps=flags.FLAGS.validation_steps,
            epochs=flags.FLAGS.epochs,
            steps_per_epoch=flags.FLAGS.steps_per_epoch,
            optimizer=optimizer,
            loss_object=loss,
            job_dir=flags.FLAGS["job-dir"].value,
        )
    else:
        train_data = train_data.map(lambda s, t: ((s, t[:, :-1]), t[:, 1:]))
        val_data = val_data.map(lambda s, t: ((s, t[:, :-1]), t[:, 1:]))

        model = tf.keras.Model(inputs=[src, tar], outputs=logits)
        model.summary()

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        model.fit(train_data,
                  validation_data=val_data,
                  validation_steps=flags.FLAGS.validation_steps,
                  epochs=flags.FLAGS.epochs,
                  steps_per_epoch=flags.FLAGS.steps_per_epoch,
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(
                          log_dir=flags.FLAGS["job-dir"].value)
                  ])

    model.save(os.path.join(flags.FLAGS["job-dir"].value, "model"))


def train(train_data, val_data, model, validation_steps, epochs,
          steps_per_epoch, optimizer, loss_object, job_dir):
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _, _, _ = model([inp, tar_inp], training=True)
            loss = loss_object(tar_real, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

        return loss

    with tf.summary.create_file_writer(job_dir).as_default():  # pylint: disable=not-context-manager
        for epoch in range(epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()

            for batch, (inp, tar) in enumerate(train_data):
                train_step(inp, tar)

                if batch % steps_per_epoch == 0:
                    logging.info('Epoch %d Batch %d Loss %.4f Accuracy %.4f',
                                 epoch + 1, batch, train_loss.result(),
                                 train_accuracy.result())

            logging.info('Epoch {} Loss {:.4f} Accuracy {:.4f}', epoch + 1,
                         train_loss.result(), train_accuracy.result())
            tf.summary.scalar("Loss", train_loss.result(), step=epoch)
            tf.summary.scalar("Accuracy", train_accuracy.result(), step=epoch)


def preprocess_dataset(dataset, tokenizer_pt, tokenizer_en, max_len):
    def encode(src, tar):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
            src.numpy()) + [tokenizer_pt.vocab_size + 1]
        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
            tar.numpy()) + [tokenizer_en.vocab_size + 1]
        return lang1, lang2

    dataset = dataset.filter(lambda src, tar: tf.logical_and(
        tf.size(src) <= max_len,
        tf.size(tar) <= max_len))

    dataset = dataset.map(lambda src, tar: tf.py_function(
        encode, [src, tar], [tf.int32, tf.int32]))

    return dataset


if __name__ == "__main__":
    print(tf.__version__)
    app.flags.DEFINE_integer("num_layers", 4, "num_layers")
    app.flags.DEFINE_integer("d_model", 128, "d_model")
    app.flags.DEFINE_integer("num_heads", 8, "num_heads")
    app.flags.DEFINE_integer("d_ff", 512, "d_ff")
    app.flags.DEFINE_float("dropout_rate", 0.1, "dropout_rate")
    app.flags.DEFINE_integer("max_len", 40, "max_len")
    app.flags.DEFINE_integer("epochs", 25, "epochs")
    app.flags.DEFINE_integer("steps_per_epoch", 250, "steps_per_epoch")
    app.flags.DEFINE_integer("validation_steps", 10, "validation_steps")
    app.flags.DEFINE_integer("batch_size", 64, "batch_size")
    app.flags.DEFINE_integer("shuffle_buffer_size", 20000,
                             "shuffle_buffer_size")
    app.flags.DEFINE_boolean("use_custom_training_loop", False,
                             "use_custom_training_loop")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string(
        "subwords_en_file", "subwords/ted_hrlr_translate/pt_to_en/subwords_en",
        "subwords_en_file")
    app.flags.DEFINE_string(
        "subwords_pt_file", "subwords/ted_hrlr_translate/pt_to_en/subwords_pt",
        "subwords_pt_file")
    app.flags.DEFINE_string("job-dir", "runs/local", "job dir")
    app.run(main)
