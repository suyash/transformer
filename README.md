# transformer

A Transformer implementation in Keras' Imperative (Subclassing) API.

The goal is to provide independent implementations of individual blocks for experimentation, especially encoder.

Currently exports `Transformer(Model)`, `Encoder(Model)`, `Decoder(Model)`, `MultiHeadAttention(Model)` and `Attention(Layer)`.

There is a program for __sentiment classification__ on imdb reviews using the `Encoder` in [imdb_sentiment.py](/imdb_sentiment.py). The script can be run on its own as it has defaults for all parameters. See the source or run with `-h` to see a list of all options.
