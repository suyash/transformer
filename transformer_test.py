"""
https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer_test.py
"""

import numpy as np
import tensorflow as tf

from transformer import Transformer


class TransformerTest(tf.test.TestCase):
    def setUp(self):
        self.t = Transformer(
            input_vocab_size=64,
            target_vocab_size=64,
            pad_id=0,
            h=4,
            d_model=64,
            d_ff=128,
            N_encoder=2,
            N_decoder=2)

        self.batch_size = 4
        self.seq_len = 5
        self.raw_input = tf.placeholder(
            tf.int32, shape=(self.batch_size, self.seq_len))
        self.fake_data = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 0, 0, 0],
            [1, 2, 3, 4, 0],
            [1, 2, 3, 0, 0],
        ])

    def test_create_padding_mask(self):
        """
        https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer_test.py#L56
        """
        with self.test_session() as sess:
            mask = sess.run(
                self.t.create_padding_mask(self.raw_input, 0),
                feed_dict={self.raw_input: self.fake_data})
            expected = np.array([
                [[1., 1., 1., 1., 1.]] * self.seq_len,
                [[1., 1., 0., 0., 0.]] * self.seq_len,
                [[1., 1., 1., 1., 0.]] * self.seq_len,
                [[1., 1., 1., 0., 0.]] * self.seq_len,
            ])
            np.testing.assert_array_equal(mask, expected)


if __name__ == '__main__':
    tf.test.main()
