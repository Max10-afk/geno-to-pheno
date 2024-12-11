fc_reg = keras.regularizers.L2(1e-1)

@keras.saving.register_keras_serializable()
class encoder(Model):
    def __init__(self, latent_dim, width = 1,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.drop = feature_drop_layer(0.2, feature_dim = 1)
        self.blocks = []
        for cur_block in range(width):
            self.blocks.append(
                [
                    #layers.Conv2D(filters = 3, kernel_size = (1, 11), data_format = "channels_first"),
                    #layers.Conv2D(filters = 1, kernel_size = (1, 1), data_format = "channels_first"),
                    #layers.MaxPool2D(pool_size = (2, 2), data_format = "channels_first"),
                    #layers.Conv2D(filters = 2, kernel_size = (2, 2), data_format = "channels_first"),
                    #layers.Conv2D(filters = 2, kernel_size = (2, 2), data_format = "channels_first"),
                    #layers.MaxPool2D(pool_size = (2, 1), data_format = "channels_first"),
                    #layers.Conv2D(filters = 1, kernel_size = (1, 1), data_format = "channels_first"),
                    #layers.Conv2D(filters = 1, kernel_size = (1, 1), data_format = "channels_first"),
                    # layers.Flatten(),
                    layers.Dense(units=latent_dim * 4, activation=None),#, kernel_regularizer=fc_reg), #, #),
                    layers.LeakyReLU(negative_slope=0.5),
                    layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 4, activation=None),#, kernel_regularizer=fc_reg), #, #),
                    layers.LeakyReLU(negative_slope=0.5),
                    layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 4, activation=None),#, kernel_regularizer=fc_reg), #, #),
                    layers.LeakyReLU(negative_slope=0.5),
                    layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 3, activation=None),#, kernel_regularizer=fc_reg), #, #),
                    layers.LeakyReLU(negative_slope=0.5),
                    layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 2, activation=None),#, kernel_regularizer=fc_reg), #, #),
                    layers.LeakyReLU(negative_slope=0.5),
                    layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim, activation=None),#, kernel_regularizer=fc_reg), #, #),
                    layers.LeakyReLU(negative_slope=0.5),
                    layers.BatchNormalization(center = True),
                ]
            )
        self.mean_dense = layers.Dense(self.latent_dim, activation=None) #kernel_regularizer = fc_reg)
        self.logvar_dense = layers.Dense(self.latent_dim, activation=None, #kernel_regularizer = fc_reg,
                                         kernel_initializer=tf.keras.initializers.Zeros())

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'width': len(self.blocks)
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Use variable arguments to simplify reconstructing the object
    
    def call(self, x, training = False):
        outputs = []
        # print(f"pre flatten x: {x.shape}")
        x = layers.Flatten()(x)
        for block in self.blocks:
            sub_x = x #self.drop(x, training = training)
            for layer in block:
                sub_x = layer(sub_x, training = training)
            outputs.append(sub_x)

        y = sum(outputs)
        mean = self.mean_dense(y)
        logvar = self.logvar_dense(y)

        return [mean, logvar]
        