import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as Backend
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras.layers import Add, Conv1D, Dense, Lambda, Layer
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
        self.dropout = 1.0 - dropout  # keep_prob
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
        out = Backend.dropout(out, self.dropout)
        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MultiHeadAttention(Model):
    """
    Multiheaded Attention by splitting across `h` attention heads
    """

    def __init__(self, d_model, h, use_mask, dropout, **kwargs):
        """
        d_model: int
        h: int
        use_mask: bool
        dropout: float[0, 1]
        """
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.use_mask = use_mask

        self.lq = Dense(d_model, activation="relu", name="query")
        self.lk = Dense(d_model, activation="relu", name="key")
        self.lv = Dense(d_model, activation="relu", name="value")

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
            name="gamma", shape=input_shape[-1:], initializer=Ones())
        self.beta = self.add_weight(
            name="beta", shape=input_shape[-1:], initializer=Zeros())
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = Backend.mean(x, axis=-1, keepdims=True)
        std = Backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderLayer(Model):
    def __init__(self, d_model, d_ff, h, dropout, **kwargs):
        """
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        """
        super(EncoderLayer, self).__init__(**kwargs)
        self.add_1 = Add()
        self.layer_norm_1 = LayerNormalization()

        self.attn = MultiHeadAttention(d_model, h, True, dropout)

        self.conv_1 = Conv1D(filters=d_ff, kernel_size=1, activation="relu")
        self.conv_2 = Conv1D(filters=d_model, kernel_size=1)

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
    def __init__(self, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, **kwargs):
        """
        N: int
        d_model: int
        d_ff: int
        h: int
        dropout: float[0, 1]
        """
        super(Encoder, self).__init__(**kwargs)
        self.encoder_layers = [
            EncoderLayer(d_model, d_ff, h, dropout) for _ in range(N)
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
