import keras
import tensorflow as tf
from keras import layers


class SelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        dropout_rate=0.0,
        name="self_attention_block",
        **kwargs
    ):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.name = name

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="self_attention_layer"
        )
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout_1")
        self.layer_norm_1 = layers.LayerNormalization(name="layer_norm_1")

        self.ff_layer = layers.Dense(key_dim, activation="relu", name="ffn_layer")

        self.dropout2 = layers.Dropout(dropout_rate)
        self.layer_norm_2 = layers.LayerNormalization(name="layer_norm_2")

    def call(self, inputs, mask=None, training=None):
        if mask is not None:
            attention_mask = tf.expand_dims(mask, axis=-1)
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = inputs + attention_output
        attention_output = self.layer_norm_1(attention_output)

        ff_output = self.ff_layer(attention_output)
        ff_output = self.dropout2(ff_output, training=training)
        ff_output = attention_output + ff_output
        ff_output = self.layer_norm_2(ff_output)

        return ff_output

    def build(self, input_shape):
        super(SelfAttentionBlock, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # Build all sub-layers
        self.attention.build(input_shape, input_shape)
        self.dropout1.build(input_shape)
        self.layer_norm_1.build(input_shape)
        self.ff_layer.build(input_shape)
        self.dropout2.build(input_shape)
        self.layer_norm_2.build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
            }
        )
        return config

    def count_params(self):
        return (
            self.attention.count_params()
            + self.ff_layer.count_params()
            + self.layer_norm_1.count_params()
            + self.layer_norm_2.count_params()
            + self.dropout1.count_params()
            + self.dropout2.count_params()
        )


class SelfAttentionStack(layers.Layer):
    def __init__(self, num_heads, key_dim, stack_size=3, dropout_rate=0.0, **kwargs):
        super(SelfAttentionStack, self).__init__(**kwargs)
        self.attention_blocks = [
            SelfAttentionBlock(
                num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate
            )
            for _ in range(stack_size)  # Example: 2 attention blocks
        ]

    def call(self, inputs, mask=None, training=None):
        x = inputs
        for block in self.attention_blocks:
            x = block(x, mask=mask, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.attention_blocks[0].num_heads,
                "key_dim": self.attention_blocks[0].key_dim,
                "stack_size": len(self.attention_blocks),
                "dropout_rate": self.attention_blocks[0].dropout_rate,
            }
        )
        return config

    def build(self, input_shape):
        super(SelfAttentionStack, self).build(input_shape)
        for block in self.attention_blocks:
            block.build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


class MultiHeadAttentionBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, **kwargs):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="multi_head_attention_layer"
        )
        self.dropout = layers.Dropout(dropout_rate, name="multi_head_attention_dropout")
        self.layer_norm = layers.LayerNormalization(
            name="multi_head_attention_layer_norm"
        )
        self.ff_layer = layers.Dense(key_dim, activation="relu", name="ffn_layer")
        self.ff_dropout = layers.Dropout(dropout_rate, name="ffn_dropout")
        self.ff_layer_norm = layers.LayerNormalization(name="ffn_layer_norm")

    def build(self, input_shape):
        super(MultiHeadAttentionBlock, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # Build all sub-layers
        self.attention.build(input_shape, input_shape)
        self.dropout.build(input_shape)
        self.layer_norm.build(input_shape)
        self.ff_layer.build(input_shape)
        self.ff_dropout.build(input_shape)
        self.ff_layer_norm.build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(
        self,
        query,
        value,
        key=None,
        key_mask=None,
        value_mask=None,
        attention_mask=None,
        training=None,
    ):
        if key is None:
            key = query

        if attention_mask is not None:
            if attention_mask.shape.rank == 2:
                attention_mask = tf.expand_dims(attention_mask, axis=-1)
        attention_output = self.attention(
            query,
            value,
            key,
            attention_mask=attention_mask,
            key_mask=key_mask,
            value_mask=value_mask,
        )
        attention_output = self.dropout(attention_output, training=training)
        attention_output = self.layer_norm(attention_output + query)
        ff_output = self.ff_layer(attention_output)
        ff_output = self.ff_dropout(ff_output, training=training)
        ff_output = self.ff_layer_norm(ff_output + attention_output)
        return ff_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
            }
        )
        return config

    def count_params(self):
        return (
            self.attention.count_params()
            + self.ff_layer.count_params()
            + self.layer_norm.count_params()
            + self.dropout.count_params()
            + self.ff_dropout.count_params()
            + self.ff_layer_norm.count_params()
        )


class PoolingAttentionBlock(layers.Layer):
    def __init__(
        self, key_dim, num_heads, num_seed_vectors, dropout_rate=0.0, **kwargs
    ):
        super(PoolingAttentionBlock, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_seed_vectors = num_seed_vectors
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="pooling_attention_layer"
        )
        self.dropout = layers.Dropout(dropout_rate, name="pooling_attention_dropout")
        self.layer_norm = layers.LayerNormalization(name="pooling_attention_layer_norm")
        self.seed_vectors = self.add_weight(
            shape=(num_seed_vectors, key_dim),
            initializer="random_normal",
            trainable=True,
            name="seed_vectors",
        )
        self.input_ff = layers.Dense(key_dim, activation="relu", name="input_ff_layer")
        self.norm_layer_1 = layers.LayerNormalization(name="norm_layer_1")
        self.output_ff = layers.Dense(
            key_dim, activation="relu", name="output_ff_layer"
        )
        self.norm_layer_2 = layers.LayerNormalization(name="norm_layer_2")

    def build(self, input_shape):
        super(PoolingAttentionBlock, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # Build all sub-layers
        self.attention.build(input_shape, input_shape)
        self.input_ff.build(input_shape)
        self.output_ff.build(input_shape)
        self.layer_norm.build(input_shape)
        self.norm_layer_1.build(input_shape)
        self.norm_layer_2.build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, inputs, mask=None, training=None):
        # inputs shape: (batch_size, sequence_length, feature_dim)
        # seed_vectors shape: (num_seed_vectors, key_dim)
        seed_vectors = tf.tile(
            tf.expand_dims(self.seed_vectors, axis=0), [tf.shape(inputs)[0], 1, 1]
        )

        attention_output = self.attention(
            seed_vectors, inputs, key_mask = mask
        )
        attention_output = self.dropout(attention_output, training=training)
        attention_output = self.layer_norm(attention_output + seed_vectors)

        ff_input = self.input_ff(attention_output)
        ff_input = self.norm_layer_1(ff_input + attention_output)

        ff_output = self.output_ff(ff_input)
        ff_output = self.norm_layer_2(ff_output + ff_input)

        return ff_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_seed_vectors, self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "num_seed_vectors": self.num_seed_vectors,
            }
        )
        return config

    def count_params(self):
        return (
            self.attention.count_params()
            + self.input_ff.count_params()
            + self.output_ff.count_params()
            + self.layer_norm.count_params()
            + self.norm_layer_1.count_params()
            + self.norm_layer_2.count_params()
            + tf.reduce_prod(self.seed_vectors.shape)
        )


class PointTransformerFromCoords(layers.Layer):
    def __init__(
        self, feature_dim, pos_mlp_hidden_dim=32, attn_mlp_hidden_dim=32, **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.pos_mlp_hidden_dim = pos_mlp_hidden_dim
        self.attn_mlp_hidden_dim = attn_mlp_hidden_dim

        # Feature embedding from coordinates
        self.coord_embed = keras.Sequential(
            [layers.Dense(feature_dim, activation="relu"), layers.Dense(feature_dim)],
            name="coord_embed",
        )

        # Linear projections
        self.linear_q = layers.Dense(feature_dim, use_bias=False)
        self.linear_k = layers.Dense(feature_dim, use_bias=False)
        self.linear_v = layers.Dense(feature_dim, use_bias=False)

        # Position encoding MLP
        self.pos_mlp = keras.Sequential(
            [
                layers.Dense(pos_mlp_hidden_dim, activation="relu"),
                layers.Dense(feature_dim),
            ],
            name="pos_mlp",
        )

        # Attention MLP
        self.attn_mlp = keras.Sequential(
            [
                layers.Dense(attn_mlp_hidden_dim, activation="relu"),
                layers.Dense(feature_dim),
            ],
            name="attn_mlp",
        )

        self.linear_out = layers.Dense(feature_dim)

    def build(self, input_shape):
        super().build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)


    def compute_output_shape(self, *args, **kwargs):
        return (None, None, self.feature_dim)

    def call(self, coords, mask=None):
        """
        coords: Tensor of shape (B, N, 3)
        mask:   Tensor of shape (B, N), boolean (True = valid point)
        """
        # Initial point feature embedding from coordinates
        x = self.coord_embed(coords)  # (B, N, C)

        # Query, key, value projections
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Compute relative positions and encode
        rel_pos = tf.expand_dims(coords, axis=2) - tf.expand_dims(
            coords, axis=1
        )  # (B, N, N, 3)
        pos_enc = self.pos_mlp(rel_pos)  # (B, N, N, C)

        # Compute attention
        q_exp = tf.expand_dims(q, axis=2)  # (B, N, 1, C)
        k_exp = tf.expand_dims(k, axis=1)  # (B, 1, N, C)
        attn = q_exp - k_exp + pos_enc  # (B, N, N, C)
        attn = self.attn_mlp(attn)  # (B, N, N, C)

        if mask is not None:
            # Expand and broadcast mask: (B, N) → (B, 1, N, 1)
            attn_mask = tf.expand_dims(mask, axis=1)  # (B, 1, N)
            attn_mask = tf.expand_dims(attn_mask, axis=-1)  # (B, 1, N, 1)
            large_neg = -1e9
            attn = tf.where(
                tf.cast(attn_mask, tf.bool), attn, large_neg * tf.ones_like(attn)
            )

        # Softmax over neighbors
        attn = tf.nn.softmax(attn, axis=2)

        # Attend over values with position encoding
        v_exp = tf.expand_dims(v, axis=1)  # (B, 1, N, C)
        output = tf.reduce_sum(attn * (v_exp + pos_enc), axis=2)  # (B, N, C)

        # Output projection + residual
        return self.linear_out(output) + x

    def get_config(self):
        return {
            **super().get_config(),
            "feature_dim": self.feature_dim,
            "pos_mlp_hidden_dim": self.pos_mlp_hidden_dim,
            "attn_mlp_hidden_dim": self.attn_mlp_hidden_dim,
        }
