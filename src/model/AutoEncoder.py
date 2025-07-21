import keras
import tensorflow as tf
import numpy as np

class AutoEncoder(keras.Model):
    def __init__(self, input_size : int = 32, latent_dim : int = 8, nodes : int | list[int] = 3, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.input_size = input_size
        self.latent_dim = latent_dim
        if isinstance(nodes, int):
            nodes = [int( input_size -  (input_size - latent_dim) / (nodes + 1) * (i + 1)) for i in range(nodes)]
        self.nodes = nodes
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(shape=(input_size,)),
            *[keras.layers.Dense(node, activation='relu') for node in nodes],
            keras.layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(shape=(latent_dim,)),
            *[keras.layers.Dense(node, activation='relu') for node in reversed(nodes)],
            keras.layers.Dense(input_size, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded