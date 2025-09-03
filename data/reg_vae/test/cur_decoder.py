@keras.saving.register_keras_serializable()
class decoder(Model):
    def __init__(self, latent_dim, depth = 1,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.depth = depth
        # self.drop = feature_drop_layer(0.2, feature_dim = 1)
        self.drop = tf.keras.layers.Dropout(0.1)
        self.blocks = []
        self.act_layer = tf.keras.layers.LeakyReLU(alpha = 0.5)
        for cur_width in range(11):
            # Create a block of layers
            block = []
            for cur_depth in range(1, self.depth):
                block.append(layers.Dense(units=50 * cur_depth,
                    name = f"dense_d_{cur_depth}_w_{cur_width}"))
                # block.append(layers.BatchNormalization(center = True,
                #     name = f"batch_norm_d_{cur_depth}_w_{cur_width}"))
                # block.append(self.drop)
            block.append(layers.Dense(units=1144))
            block.append(self.act_layer)
            self.blocks.append(block)

    def get_config(self):
        config = super(decoder, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'depth': self.depth
        })
        return config
    #def build(self, input_shape):
    #    # Determine the shape after embedding
    #    batch_size = input_shape[0][0]  # Assuming the first dimension of parents gives the batch size
    #    latent_input_shape = [batch_size, self.latent_dim]
    #
    #    # You don't need to build layers explicitly unless for particular reasons,
    #    # as Keras does that during the first call.
    #    # But here's how you'd do it manually for immediate setup:
    #    for block in self.blocks:
    #        for layer in block:
    #            layer.build(latent_input_shape)
    #            if isinstance(layer, layers.Dense):
    #                latent_input_shape = [batch_size, layer.units]
    #
    #    # Mark the model as built
    #    super().build(input_shape)


    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
    def call(self, parents, embed_x, training = False, return_activations = False):
        act_tracker = {}
        outputs = []
        # embed_x = self.drop(embed_x, training = training)
        for block in self.blocks:
            sub_x = embed_x
            for layer in block:
                sub_x = layer(sub_x, training = training)
                sub_x = self.act_layer(sub_x)
                act_tracker[layer.name] = tf.reduce_mean(tf.reshape(sub_x, [sub_x.shape[0], -1]), axis = 1)
            sub_x = tf.reshape(sub_x, (-1, 1144, 1))
            outputs.append(sub_x)
        y = tf.concat(outputs, axis=-1)

        # Sum up parent vectors over feature axis
        prod_parents = tf.math.reduce_sum(parents, axis = 1)
        y = prod_parents + y
        if return_activations:
            return y, act_tracker
        else:
            return y, {}
