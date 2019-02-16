# transformer

A Transformer implementation in Keras' Imperative (Subclassing) API.

The goal is to provide independent implementations of individual blocks for experimentation, especially encoder.

Currently exports `Transformer(Model)`, `Encoder(Model)`, `Decoder(Model)`, `MultiHeadAttention(Model)` and `Attention(Layer)`.

There is a program for __sentiment classification__ on imdb reviews using the `Encoder` in [imdb_sentiment.py](/imdb_sentiment.py). The script can be run on its own as it has defaults for all parameters. See the source or run with `-h` to see a list of all options.

__Currently the models cannot be saved after training, however model weights can be.__ Keras does not allow imperative models to be saved without implementing manual serialization, however there does not seem to be a way to do that. [TensorFlow's next release will have support for saving](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/contrib/saved_model/save_keras_model), but [unable to get it to work in current rc1 release](https://colab.research.google.com/gist/suyash/de8c6a386ff3a18c499aa441d5a13670).
