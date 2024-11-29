@keras.saving.register_keras_serializable()
class decoder(Model):
    def __init__(self, latent_dim, depth = 1,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.depth = depth
        self.drop = feature_drop_layer(0.1, feature_dim = 1)
        self.blocks = []
        # self.drop = layers.SpatialDropout1D(0.5)
        for _ in range(11):
            # Create a block of layers
            block = []    
            # Add dense layers with Leaky ReLU activation after each
            for i in range(1, self.depth):
                block.append(layers.Dense(units=50 * i))
                layers.BatchNormalization(center = False)
                block.append(layers.LeakyReLU(negative_slope=0.5))
            # Add a final dense layer with fixed output units and kernel regularization
            block.append(layers.Dense(units=1144))
            block.append(layers.LeakyReLU(negative_slope=0.5))  # Add Leaky ReLU after the final dense layer
            # Append the block to the blocks list
            self.blocks.append(block)

    def get_config(self):
        config = super(decoder, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'depth': self.depth
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
    def call(self, parents, embed_x, training = False):

        outputs = []
        # embed_x = self.drop(embed_x, training = training)
        for block in self.blocks:
            sub_x = embed_x
            for layer in block:
                sub_x = layer(sub_x, training = training)
            sub_x = tf.reshape(sub_x, (-1, 1144, 1))
            outputs.append(sub_x)
        y = tf.concat(outputs, axis=-1)
        prod_parents = tf.math.reduce_sum(parents, axis = 1)
        #prod_parents = self.drop(prod_parents, training = training)
        # print(f"prod_parents.shape: {prod_parents.shape}")
        # print(f"training: {training}")
        y = prod_parents + y

        return y
