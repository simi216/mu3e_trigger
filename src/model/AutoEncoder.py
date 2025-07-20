import keras
import tensorflow as tf
import numpy as np

class AutoEncoder(keras.Model):
    def __init__(self, nodes :list[int]= [32,16, 8], latent_dim : int = 8, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.nodes = nodes
        self.latent_dim = latent_dim
        nodes = np.array(nodes)
        self.encoder = keras.Sequential(
            [keras.layers.InputLayer(input_shape=(nodes[0],), name="input_layer")] +
            [keras.layers.Dense(node_dim,  activation="relu") for node_dim in nodes], 
            name="encoder")

        self.decoder = keras.Sequential(
            [keras.layers.InputLayer(input_shape=(latent_dim,), name="latent_input_layer")] +
             [keras.layers.Dense(node_dim, activation="relu") for node_dim in nodes[::-1]],
            name="decoder")

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded