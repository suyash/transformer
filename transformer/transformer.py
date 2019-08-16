import math

import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Layer, LayerNormalization, Multiply, Permute, Reshape  # pylint: disable=import-error


def gelu(x, faster_approx=False):
    """
    - https://arxiv.org/abs/1606.08415
    - https://github.com/google-research/bert/blob/master/modeling.py#L264-L277
    - https://github.com/hendrycks/GELUs
    """

    if faster_approx:
        cdf = tf.sigmoid(1.072 * x)
    else:
        cdf = 0.5 * (1.0 + tf.tanh(
            (tf.math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))

    return x * cdf


class Transformer:
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 input_vocab_size,
                 target_vocab_size,
                 dropout_rate,
                 ffn_activation=tf.keras.activations.relu,
                 scope="transformer"):
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate,
                               ffn_activation=ffn_activation,
                               scope="%s/encoder" % scope)

        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate,
                               ffn_activation=ffn_activation,
                               scope="%s/decoder" % scope)

        self.final_layer = Dense(target_vocab_size,
                                 activation=None,
                                 name="%s/dense" % scope)

        self.padding_mask = PaddingMask(name="%s/padding_mask" % scope)
        self.lookahead_mask = PaddingAndLookaheadMask(
            name="%s/lookahead_mask" % scope)

    def __call__(self, inputs, target):
        padding_mask = self.padding_mask(inputs)
        lookahead_mask = self.lookahead_mask(target)

        enc_output, enc_attention = self.encoder(inputs, padding_mask)

        dec_output, dec_attention, enc_dec_attention = self.decoder(
            target, enc_output, lookahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, enc_attention, dec_attention, enc_dec_attention


class Decoder:
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 vocab_size,
                 dropout_rate,
                 ffn_activation=tf.keras.activations.relu,
                 scope="decoder"):
        self.d_model = d_model
        self.num_layers = num_layers
        self.scope = scope

        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=d_model,
                                   name="%s/embedding" % scope)
        self.pos_encoding = PositionalEncoding(d_model,
                                               name="%s/positional_encoding" %
                                               scope)

        self.dec_layers = [
            DecoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout_rate=dropout_rate,
                         ffn_activation=ffn_activation,
                         scope="%s/decoder_layer_%d" % (scope, i))
            for i in range(num_layers)
        ]

        self.dropout = Dropout(dropout_rate, name="%s/dropout" % self.scope)

    def __call__(self, x, enc_output, lookahead_mask, padding_mask):
        x = self.embedding(x)
        x = MultiplyConstant(self.d_model, name="%s/multiply" % self.scope)(x)
        x = Add(name="%s/add" % self.scope)([x, self.pos_encoding(x)])

        dec_attention_weights = {}
        enc_dec_attention_weights = {}

        for i in range(self.num_layers):
            x, dec_attention, enc_dec_attention = self.dec_layers[i](
                x, enc_output, lookahead_mask, padding_mask)

            dec_attention_weights["layer_%d" % i] = dec_attention
            enc_dec_attention_weights["layer_%d" % i] = enc_dec_attention

        return x, dec_attention_weights, enc_dec_attention_weights


class Encoder:
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 vocab_size,
                 dropout_rate,
                 ffn_activation=tf.keras.activations.relu,
                 scope="encoder"):
        self.d_model = d_model
        self.num_layers = num_layers
        self.scope = scope

        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=d_model,
                                   name="%s/embedding" % scope)
        self.pos_encoding = PositionalEncoding(d_model,
                                               name="%s/positional_encoding" %
                                               scope)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout_rate=dropout_rate,
                         ffn_activation=ffn_activation,
                         scope="%s/encoder_layer_%d" % (scope, i))
            for i in range(num_layers)
        ]

        self.dropout = Dropout(dropout_rate, name="%s/dropout" % self.scope)

    def __call__(self, x, padding_mask):
        x = self.embedding(x)
        x = MultiplyConstant(self.d_model, name="%s/multiply" % self.scope)(x)
        x = Add(name="%s/add" % self.scope)([x, self.pos_encoding(x)])

        enc_attention_weights = {}

        for i in range(self.num_layers):
            x, enc_attention = self.enc_layers[i](x, padding_mask)
            enc_attention_weights["layer_%d" % i] = enc_attention

        return x, enc_attention_weights


class DecoderLayer:
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 ffn_activation=tf.keras.activations.relu,
                 scope="decoder_layer"):
        self.scope = scope

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads,
                                       scope="%s/multi_head_attention_1" %
                                       scope)
        self.mha2 = MultiHeadAttention(d_model,
                                       num_heads,
                                       scope="%s/multi_head_attention_2" %
                                       scope)
        self.ffn = PointwiseFeedForwardNetwork(
            d_model,
            d_ff,
            activation=ffn_activation,
            scope="%s/pointwise_feed_forward_network" % scope)

        self.layernorm1 = LayerNormalization(epsilon=1e-6,
                                             name="%s/layer_norm_1" % scope)
        self.layernorm2 = LayerNormalization(epsilon=1e-6,
                                             name="%s/layer_norm_2" % scope)
        self.layernorm3 = LayerNormalization(epsilon=1e-6,
                                             name="%s/layer_norm_3" % scope)

        self.dropout1 = Dropout(dropout_rate, name="%s/dropout_1" % scope)
        self.dropout2 = Dropout(dropout_rate, name="%s/dropout_2" % scope)
        self.dropout3 = Dropout(dropout_rate, name="%s/dropout_3" % scope)

    def __call__(self, x, enc_output, lookahead_mask, padding_mask):
        out1, dec_dec_attention = self.mha1(x, x, x, lookahead_mask)
        out1 = self.dropout1(out1)
        x = Add(name="%s/add_1" % self.scope)([x, out1])
        x = self.layernorm1(x)

        out2, enc_dec_attention = self.mha2(x, enc_output, enc_output,
                                            padding_mask)
        out2 = self.dropout2(out2)
        x = Add(name="%s/add_2" % self.scope)([x, out2])
        x = self.layernorm2(x)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output)
        x = Add(name="%s/add_3" % self.scope)([x, ffn_output])
        x = self.layernorm3(x)

        return x, dec_dec_attention, enc_dec_attention


class EncoderLayer:
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 ffn_activation=tf.keras.activations.relu,
                 scope="encoder_layer"):
        self.scope = scope

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads,
                                       scope="%s/multi_head_attention_1" %
                                       scope)
        self.ffn = PointwiseFeedForwardNetwork(
            d_model,
            d_ff,
            activation=ffn_activation,
            scope="%s/pointwise_feed_forward_network" % scope)

        self.layernorm1 = LayerNormalization(epsilon=1e-6,
                                             name="%s/layer_norm_1" % scope)
        self.layernorm2 = LayerNormalization(epsilon=1e-6,
                                             name="%s/layer_norm_2" % scope)

        self.dropout1 = Dropout(dropout_rate, name="%s/dropout_1" % scope)
        self.dropout2 = Dropout(dropout_rate, name="%s/dropout_2" % scope)

    def __call__(self, x, padding_mask):
        out1, enc_enc_attention = self.mha1(x, x, x, padding_mask)
        out1 = self.dropout1(out1)
        x = Add(name="%s/add_1" % self.scope)([x, out1])
        x = self.layernorm1(x)

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = Add(name="%s/add_2" % self.scope)([x, ffn_output])
        x = self.layernorm2(x)

        return x, enc_enc_attention


class PointwiseFeedForwardNetwork:
    def __init__(self,
                 d_model,
                 d_ff,
                 activation=tf.keras.activations.relu,
                 scope="pointwise_feed_forward_network"):
        self.dense_1 = Dense(d_ff,
                             activation=activation,
                             name="%s/dense_1" % scope)
        self.dense_2 = Dense(d_model,
                             activation=None,
                             name="%s/dense_2" % scope)

    def __call__(self, x):
        return self.dense_2(self.dense_1(x))


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, scope="multi_head_attention"):
        assert d_model % num_heads == 0

        self.wq = Dense(d_model, name="%s/dense_q" % scope)
        self.wk = Dense(d_model, name="%s/dense_k" % scope)
        self.wv = Dense(d_model, name="%s/dense_v" % scope)

        self.reshapeq = Reshape((-1, num_heads, d_model // num_heads),
                                name="%s/reshape_q" % scope)
        self.reshapek = Reshape((-1, num_heads, d_model // num_heads),
                                name="%s/reshape_k" % scope)
        self.reshapev = Reshape((-1, num_heads, d_model // num_heads),
                                name="%s/reshape_v" % scope)

        self.transposeq = Permute((2, 1, 3), name="%s/transpose_q" % scope)
        self.transposek = Permute((2, 1, 3), name="%s/transpose_k" % scope)
        self.transposev = Permute((2, 1, 3), name="%s/transpose_v" % scope)

        self.reshape_output = Reshape((-1, d_model),
                                      name="%s/reshape_output" % scope)

        self.transpose_output = Permute((2, 1, 3),
                                        name="%s/transpose_output" % scope)

        self.dense = Dense(d_model, name="%s/dense" % scope)

        self.attention = Attention(name="%s/attention" % scope)

    def __call__(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.reshapeq(q)
        k = self.reshapek(k)
        v = self.reshapev(v)

        q = self.transposeq(q)
        k = self.transposek(k)
        v = self.transposev(v)

        x, attention_weights = self.attention([q, k, v, mask])

        x = self.transpose_output(x)
        x = self.reshape_output(x)
        x = self.dense(x)

        return x, attention_weights


class Attention(Layer):
    def call(self, inputs):
        q, k, v, mask = inputs

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights


class PositionalEncoding(Layer):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model

    def call(self, inputs):
        position = tf.shape(inputs)[1]

        position_dims = tf.range(position)[:, tf.newaxis]
        embed_dims = tf.range(self.d_model)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(
            10000.0, tf.cast(
                (2 * (embed_dims // 2)) / self.d_model, tf.float32))
        angle_rads = tf.cast(position_dims, tf.float32) * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_config(self):
        base = super().get_config()
        return dict(list(base.items()) + [("d_model", self.d_model)])


class MultiplyConstant(Layer):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def call(self, inputs):
        return inputs * self.c

    def get_config(self):
        base = super().get_config()
        return dict(list(base.items()) + [("c", self.c)])


class PaddingMask(Layer):
    def call(self, inputs):
        seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]


class PaddingAndLookaheadMask(Layer):
    def call(self, inputs):
        size = tf.shape(inputs)[1]
        lhm = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        seq = seq[:, tf.newaxis, tf.newaxis, :]

        return tf.maximum(lhm, seq)
