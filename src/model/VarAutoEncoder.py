import keras
import tensorflow as tf
import numpy as np

class VarAutoEncoder(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(VarAutoEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(latent_dim + latent_dim),  # Mean and log variance
        ])
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(7 * 7 * 64, activation='relu'),
            keras.layers.Reshape((7, 7, 64)),
            keras.layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'),
            keras.layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same'),
        ])

    def encode(self, x):
        z_mean_log_var = self.encoder(x)
        z_mean, z_log_var = tf.split(z_mean_log_var, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed