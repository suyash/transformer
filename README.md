# transformer

A Transformer implementation in Keras' Imperative (Subclassing) API.

The goal is to provide independent implementations of individual blocks for experimentation, especially encoder.

Currently exports `Transformer(Model)`, `Encoder(Model)`, `Decoder(Model)`, `MultiHeadAttention(Model)` and `Attention(Layer)`.

There is a program for __sentiment classification__ on imdb reviews using the `Encoder` in [imdb_sentiment.py](/imdb_sentiment.py). The script can be run on its own as it has defaults for all parameters. See the source or run with `-h` to see a list of all options.

__Currently the models cannot be saved after training, however model weights can be.__ Keras does not allow imperative models to be saved without implementing manual serialization, however there does not seem to be a way to do that. [TensorFlow's next release](https://github.com/tensorflow/tensorflow/blob/r1.13/RELEASE.md#bug-fixes-and-other-changes) [will have support for saving](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/contrib/saved_model/save_keras_model), but [unable to get it to work in current rc2 release](https://colab.research.google.com/gist/suyash/de8c6a386ff3a18c499aa441d5a13670).

The `imdb_sentiment` example exports a `create_model` function to create the keras model instance used for training, so doing

```py
from imdb_sentiment import create_model

model = create_model(...)
model.load_weights(weights_dir)
```

still works.

[Relevant StackOverflow Thread](https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model)

## TODO

- Custom Adam Implementation with learning rate schedule as described in the paper.

- byte-pair encoding for translation inputs. Currently, with default vocab on wmt14, the model gets to 107,248,227 parameters, which is significantly higher than the paper which reports 65M parameters for the base model.
