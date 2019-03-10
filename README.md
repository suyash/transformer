# transformer

A Transformer implementation in Keras' Imperative (Subclassing) API.

The goal is to provide independent implementations of individual blocks for experimentation, especially encoder.

Currently exports `Transformer(Model)`, `Encoder(Model)`, `Decoder(Model)`, `MultiHeadAttention(Model)` and `Attention(Layer)`.

There is a program for __sentiment classification__ on imdb reviews using the `Encoder` in [imdb_sentiment.py](/imdb_sentiment.py). The script can be run on its own as it has defaults for all parameters. See the source or run with `-h` to see a list of all options.

__On master, the models cannot be saved after training, however model weights can be.__ TensorFlow 1.13 does introduce model saving for keras imperative models, but that requires eager, which was giving other errors for me. To save models, I just did estimators on the `non-subclassing` branch.

The `imdb_sentiment` example exports a `create_model` function to create the keras model instance used for training, so doing

```py
from imdb_sentiment import create_model

model = create_model(...)
model.load_weights(weights_dir)
```

still works.

The [__non-subclassing__](https://github.com/suyash/transformer/tree/non-subclassing) branch removes subclassing to build a regular model, without any abstractions. This allows creating an estimator for training. The imdb demo notebook uses a model [trained on that branch](https://github.com/suyash/transformer/blob/non-subclassing/imdb_sentiment.py#L84).

The [`imdb_sentiment_demo`](https://colab.research.google.com/github/suyash/transformer/blob/master/imdb_sentiment_demo.ipynb) notebook also contains heatmaps for different attention heads for different layers. Couldn't get a visualization similar to [Tensor2Tensor](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) to work.

There is also a script `translation.py` for trying the model on different NMT tasks. Trained wmt_en_de_2014 for 1_000 steps, will train one for a longer time later.

## TODO

- Custom Adam Implementation with learning rate schedule as described in the paper.

- Using an actual tokenizer to preprocess imdb data instead of just splitting on spaces.

- byte-pair encoding for translation inputs. Currently, with default vocab on wmt_en_de_2014, the model gets to 107,248,227 parameters, which is significantly higher than the paper which reports 65M parameters for the base model.
