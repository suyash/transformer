"""
http://nlp.seas.harvard.edu/2018/04/03/attention.html

+

https://github.com/lilianweng/transformer-tensorflow
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as Backend
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Add, Conv1D, Dense, Dropout, Lambda, Layer
from tensorflow.keras.models import Model


class Attention(Layer):
    """
    Scaled Dot Product Attention
    """

    def __init__(self, d_k, use_mask, dropout, **kwargs):
        """
        d_model: int
        h: int
        use_mask: bool
        dropout: float[0, 1]
        """
        self.d_k = d_k
        self.dropout = dropout
        self.use_mask = use_mask
        super(Attention, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Q: [h * batch, q_size, d_model]
        K: [h * batch, k_size, d_model]
        V: [h * batch, k_size, d_model]
        mask?: [h * batch, q_size, k_size]

        [h * batch, q_size, d_model]
        """

        Q, K, V = inputs[0], inputs[1], inputs[2]
        if self.use_mask:
            mask = inputs[3]

        out = tf.matmul(Q, tf.transpose(
            K, [0, 2, 1]))  # [h * batch, q_size, k_size]
        out = out / Backend.sqrt(Backend.cast(self.d_k, tf.float32))

        if self.use_mask:
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = Backend.softmax(out)

        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/layers/core.py#L136
        out = tf.contrib.framework.smart_cond(
            Backend.learning_phase(),
            lambda: Backend.dropout(out, self.dropout),
            lambda: tf.identity(out))

        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MultiHeadAttention(Model):
    """
    Multiheaded Attention by splitting across `h` attention heads
    """

    def __init__(self,
                 d_model,
                 h,
                 use_mask,
                 dropout,
                 initializer="glorot_uniform",
                 **kwargs):
        """
        d_model: int
        h: int
        use_mask: bool
        dropout: float[0, 1]
        """
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.use_mask = use_mask

        self.lq = Dense(
            d_model,
            activation="relu",
            name="query",
            kernel_initializer=initializer,
            bias_initializer=initializer)
        self.lk = Dense(
            d_model,
            activation="relu",
            name="key",
            kernel_initializer=initializer,
            bias_initializer=initializer)
        self.lv = Dense(
            d_model,
            activation="relu",
            name="value",
            kernel_initializer=initializer,
            bias_initializer=initializer)

        self.splitter = Lambda(
            lambda t: tf.concat(tf.split(t, h, axis=2), axis=0),
            name="split_attention_heads")
        self.joiner = Lambda(
            lambda t: tf.concat(tf.split(t, h, axis=0), axis=2),
            name="join_attention_heads")

        if use_mask:
            self.mask_expander = Lambda(
                lambda t: tf.tile(t, [h, 1, 1]), name="expand_mask")

        self.attention = Attention(d_model // h, use_mask, dropout)

    def call(self, inputs):
        """
        inputs:
            query: [batch, q_size, d_model]
            key: [batch, k_size, d_model]
            value: [batch, k_size, d_model]
            mask?: [batch, q_size, k_size]

        returns:
            [batch, q_size, d_model]
        """
        query, key, value = inputs[0], inputs[1], inputs[2]
        if self.use_mask:
            mask = inputs[3]

        Q = self.lq(query)
        K = self.lk(key)
        V = self.lv(value)

        Q_split = self.splitter(Q)
        K_split = self.splitter(K)
        V_split = self.splitter(V)

        if self.use_mask:
            mask = self.mask_expander(mask)
            out = self.attention([Q_split, K_split, V_split, mask])
        else:
            out = self.attention([Q_split, K_split, V_split])

        out = self.joiner(out)

        return out


class LayerNormalization(Layer):
    """
    https://github.com/keras-team/keras/issues/3878
    https://github.com/Lsdefine/attention-is-all-you-need-keras/blob/master/transformer.py#L14
    """

    def __init__(self, eps=1e-8, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma", shape=input_shape[-1:], initializer="ones")
        self.beta = self.add_weight(
            name="beta", shape=input_shape[-1:], initializer="zeros")
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = Backend.mean(x, axis=-1, keepdims=True)
        std = Backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class Embedding(Layer):
    """
    A custom `Embedding` layer implementation, that additionally takes a `pad_id` and keeps the embedding
    for that token fixed to zero.

    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/layers/embeddings.py
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 pad_id,
                 embeddings_initializer="glorot_uniform",
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

    def build(self, input_shape):
        self.pad_embeddings = self.add_weight(
            shape=(1, self.embedding_size),
            initializer="zeros",
            name='pad_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=False)

        embeddings = [self.pad_embeddings]

        if self.pad_id > 0:
            self.pre_pad_embeddings = self.add_weight(
                shape=(self.pad_id, self.embedding_size),
                initializer=self.embeddings_initializer,
                name='pre_pad_embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint)

            embeddings.insert(0, self.pre_pad_embeddings)

        if self.pad_id < self.vocab_size - 1:
            self.post_pad_embeddings = self.add_weight(
                shape=(self.vocab_size - self.pad_id - 1, self.embedding_size),
                initializer=self.embeddings_initializer,
                name='post_pad_embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint)

            embeddings.append(self.post_pad_embeddings)

        self.embeddings = tf.concat(embeddings, axis=0)

    def call(self, inputs):
        dtype = Backend.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = tf.cast(inputs, "int32")
        return tf.nn.embedding_lookup(self.embeddings, inputs)


class EncoderLayer(Model):
    """
    self attention + FF
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 h,
                 dropout,
                 initializer="glorot_uniform",
                 **kwargs):
        """
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        """
        super(EncoderLayer, self).__init__(**kwargs)
        self.add_1 = Add()
        self.layer_norm_1 = LayerNormalization()

        self.attn = MultiHeadAttention(
            d_model, h, True, dropout, initializer=initializer)

        self.conv_1 = Conv1D(
            filters=d_ff,
            kernel_size=1,
            activation="relu",
            kernel_initializer=initializer,
            bias_initializer=initializer)
        self.conv_2 = Conv1D(
            filters=d_model,
            kernel_size=1,
            kernel_initializer=initializer,
            bias_initializer=initializer)

        self.add_2 = Add()
        self.layer_norm_2 = LayerNormalization()

    def call(self, inputs):
        """
        inp: [batch, seq_len, d_model]
        mask: [batch, seq_len, seq_len]
        """
        inp, mask = inputs
        out = inp
        out = self.layer_norm_1(
            self.add_1([out, self.attn([out, out, out, mask])]))
        out = self.layer_norm_2(self.add_2([out, self.feed_forward(out)]))
        return out

    def feed_forward(self, out):
        """
        Positionwise FeedForward

        2 options:
        - linear + relu + linear
        - convolution + relu + convolution (kernel_size=1)

        input:
            out: [batch, seq_len, d_model]
        """
        out = self.conv_1(out)  # [batch, seq_len, d_ff]
        return self.conv_2(out)  # [batch, seq_len, d_model]


class Encoder(Model):
    """
    N `EncoderLayer`s in sequence
    """

    def __init__(self,
                 N=6,
                 d_model=512,
                 d_ff=2048,
                 h=8,
                 dropout=0.1,
                 initializer="glorot_uniform",
                 **kwargs):
        """
        N: int
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        """
        super(Encoder, self).__init__(**kwargs)
        self.encoder_layers = [
            EncoderLayer(d_model, d_ff, h, dropout, initializer=initializer)
            for _ in range(N)
        ]

    def call(self, inputs):
        """
        inp: [batch, seq_len, d_model]
        mask: [batch, seq_len, seq_len]
        """
        inp, mask = inputs
        out = inp

        for layer in self.encoder_layers:
            out = layer([out, mask])

        return out


class DecoderLayer(Model):
    """
    self attention + src attention + FF
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 h,
                 dropout,
                 initializer="glorot_uniform",
                 **kwargs):
        """
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        initializer: string/keras.initializers object
        """
        super(DecoderLayer, self).__init__(**kwargs)
        self.add_1 = Add()
        self.layer_norm_1 = LayerNormalization()

        self.attn_1 = MultiHeadAttention(
            d_model, h, True, dropout, initializer=initializer)

        self.add_2 = Add()
        self.layer_norm_2 = LayerNormalization()

        self.attn_2 = MultiHeadAttention(
            d_model, h, True, dropout, initializer=initializer)

        self.conv_1 = Conv1D(
            filters=d_ff,
            kernel_size=1,
            activation="relu",
            kernel_initializer=initializer,
            bias_initializer=initializer)
        self.conv_2 = Conv1D(
            filters=d_model,
            kernel_size=1,
            kernel_initializer=initializer,
            bias_initializer=initializer)

        self.add_3 = Add()
        self.layer_norm_3 = LayerNormalization()

    def call(self, inputs):
        """
        inp: [batch, seq_len, d_model]
        memory: [batch, seq_len, d_model]
        input_mask: [batch, seq_len, seq_len]
        target_mask: [batch, seq_len, seq_len]
        """
        inp, memory, input_mask, target_mask = inputs
        out = inp
        out = self.layer_norm_1(
            self.add_1([out, self.attn_1([out, out, out, target_mask])]))
        out = self.layer_norm_2(
            self.add_2([out,
                        self.attn_2([out, memory, memory, input_mask])]))
        out = self.layer_norm_3(self.add_3([out, self.feed_forward(out)]))
        return out

    def feed_forward(self, out):
        """
        Positionwise FeedForward

        2 options:
        - linear + relu + linear
        - convolution + relu + convolution (kernel_size=1)

        input:
            out: [batch, seq_len, d_model]
        """
        out = self.conv_1(out)  # [batch, seq_len, d_ff]
        return self.conv_2(out)  # [batch, seq_len, d_model]


class Decoder(Model):
    """
    N `DecoderLayer`s in sequence
    """

    def __init__(self,
                 N=6,
                 d_model=512,
                 d_ff=2048,
                 h=8,
                 dropout=0.1,
                 initializer="glorot_uniform",
                 **kwargs):
        """
        N: int
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        initializer: string/keras.initializers object
        """
        super(Decoder, self).__init__(**kwargs)
        self.decoder_layers = [
            DecoderLayer(d_model, d_ff, h, dropout, initializer=initializer)
            for _ in range(N)
        ]

    def call(self, inputs):
        """
        inp: [batch, seq_len, d_model]
        enc_out: encoder output [batch, seq_len, d_model]
        input_mask: [batch, seq_len, seq_len]
        target_mask: [batch, seq_len, seq_len]
        """
        inp, enc_out, input_mask, target_mask = inputs
        out = inp

        for layer in self.decoder_layers:
            out = layer([out, enc_out, input_mask, target_mask])

        return out


class Transformer(Model):
    """
    input embedding + encoder -> target_embedding + decoder -> logits

    This simply takes of inputs of shape [batch_size, input_seq_len], [batch_size, target_seq_len]
    and will give logits of shape [batch_size, target_seq_len, target_vocab_size].
    Reduction, loss calculation etc. is left to the caller
    """

    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 pad_id,
                 N_encoder=6,
                 N_decoder=6,
                 d_model=512,
                 d_ff=2048,
                 h=8,
                 dropout=0.1,
                 initializer="glorot_uniform",
                 **kwargs):
        """
        input_vocab_size: int
        target_vocab_size: int
        pad_id: int
        N_encoder: int
        N_decoder: int
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        initializer: string/keras.initializers object
        """
        super(Transformer, self).__init__(**kwargs)

        self.d_model = d_model
        self.dropout = dropout

        self.encoder = Encoder(
            N_encoder, d_model, d_ff, h, dropout, initializer=initializer)
        self.decoder = Decoder(
            N_decoder, d_model, d_ff, h, dropout, initializer=initializer)
        self.input_mask_layer = Lambda(
            lambda t: create_padding_mask(t, pad_id), name="input_mask")
        self.target_mask_layer = Lambda(
            lambda t: create_padding_mask(t, pad_id) * create_subsequent_mask(t),
            name="target_mask")
        self.logits_layer = Dense(
            target_vocab_size,
            name="logits",
            kernel_initializer=initializer,
            bias_initializer=initializer)

        self.inp_embed = Embedding(
            input_vocab_size,
            d_model,
            pad_id,
            embeddings_initializer=initializer)
        self.tar_embed = Embedding(
            target_vocab_size,
            d_model,
            pad_id,
            embeddings_initializer=initializer)

        self.positional_encode = Lambda(
            lambda t: positional_encoding(t, d_model),
            name="positional_encoding")

    def call(self, inputs):
        """
        enc_inp: [batch_size, seq_len]
        dec_inp: [batch_size, seq_len]
        """
        enc_inp, dec_inp = inputs

        inp_mask = self.input_mask_layer(enc_inp)
        tar_mask = self.target_mask_layer(dec_inp)

        enc_inp = Dropout(self.dropout)(
            Add()([self.inp_embed(enc_inp),
                   self.positional_encode(enc_inp)]))
        dec_inp = Dropout(self.dropout)(
            Add()([self.tar_embed(dec_inp),
                   self.positional_encode(dec_inp)]))

        enc_out = self.encoder([enc_inp, inp_mask])
        dec_out = self.decoder([dec_inp, enc_out, inp_mask, tar_mask])

        logits = self.logits_layer(dec_out)
        return logits


def create_padding_mask(inp, pad_id):
    """
    creates an attention mask that blocks current token from attending to padding tokens

    inp: [batch_size, seq_len]
    pad_id: int

    returns: [batch_size, seq_len, seq_len]
    """
    mask = tf.cast(tf.not_equal(inp, pad_id), tf.float32)
    mask = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(inp)[1], 1])
    return mask


def create_subsequent_mask(inp):
    """
    creates an attention mask that blocks current token from attending to tokens after itself

    https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer.py#L380

    inp: [batch_size, seq_len]

    returns: [batch_size, seq_len, seq_len]
    """
    seq_len = inp.shape.as_list()[1]

    tri_matrix = np.zeros((seq_len, seq_len))
    tri_matrix[np.tril_indices(seq_len)] = 1

    mask = tf.convert_to_tensor(tri_matrix, dtype=tf.float32)
    mask = tf.tile(tf.expand_dims(mask, 0), [tf.shape(inp)[0], 1, 1])
    return mask


def positional_encoding(inp, d_model):
    """
    sinusoidal positional encoding for the inputs

    inp: [batch_size, seq_len]

    returns: [batch_size, seq_len, d_model]
    """
    seq_len = inp.shape.as_list()[1]

    encoding = np.array([[
        pos / np.power(10000.0, 2.0 * (i // 2) / d_model)
        for i in range(d_model)
    ] for pos in range(seq_len)])
    encoding[:, 0::2] = np.sin(encoding[:, 0::2])
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])

    encoding = tf.convert_to_tensor(encoding, dtype=tf.float32)
    return tf.tile(tf.expand_dims(encoding, 0), [tf.shape(inp)[0], 1, 1])
