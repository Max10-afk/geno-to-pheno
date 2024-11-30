fc_reg = keras.regularizers.L2(1e-1)
@keras.saving.register_keras_serializable()
class decoder(Model):
    def __init__(self, latent_dim, depth = 1,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.depth = depth
        self.drop = feature_drop_layer(0.2, feature_dim = 1)
        self.blocks = []

        for _ in range(11):
            # Create a block of layers
            block = []
            for i in range(1, self.depth):
                block.append(layers.Dense(units=50 * i)),#, kernel_regularizer = fc_reg))
                layers.BatchNormalization(center = False)
                block.append(layers.LeakyReLU(negative_slope=0.5))
            block.append(layers.Dense(units=1144)),#, kernel_regularizer = fc_reg))
            block.append(layers.LeakyReLU(negative_slope=0.5))
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
        embed_x = self.drop(embed_x, training = training)
        for block in self.blocks:
            sub_x = embed_x
            for layer in block:
                sub_x = layer(sub_x, training = training)
            sub_x = tf.reshape(sub_x, (-1, 1144, 1))
            outputs.append(sub_x)
        y = tf.concat(outputs, axis=-1)

        # Sum up parent vectors over feature axis
        prod_parents = tf.math.reduce_sum(parents, axis = 1)
        y = prod_parents + y

        return y
