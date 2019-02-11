import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as Backend


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
        self.dropout = 1.0 - dropout  # keep_prob

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
