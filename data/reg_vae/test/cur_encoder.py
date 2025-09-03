@keras.saving.register_keras_serializable()
class encoder(Model):
    def __init__(self, latent_dim, width = 1,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        # self.drop = feature_drop_layer(0.2, feature_dim = 2)
        self.drop = tf.keras.layers.Dropout(0.2)
        self.blocks = []
        self.act_layer = ACT_LAYER
        # num_conv_iterations = 5
        # self.conv_layers = []
        # conv_kernal_size = 2
        # self.pool_step = tf.keras.layers.MaxPooling2D(pool_size=[3, 1],
        #     data_format = "channels_first")
        # for i in range(num_conv_iterations):
        #     filter_size = 20 + 44 * (i + 1)
        #     self.conv_layers.append(
        #         tf.keras.layers.Conv2D(
        #             filter_size,
        #             kernel_size=[2, 1],
        #             activation=None,
        #             name="CONV_" + str(i),
        #             data_format = "channels_first"
        #         )
        #     )
        #     self.conv_layers[-1].name = f"conv_{i}"
        for cur_width in range(width):
            self.blocks.append(
                [
                    layers.Dense(units=latent_dim * 4, activation=None,
                        name = f"dense_d_1_w_{cur_width}"),#, kernel_regularizer=fc_reg)
                    # self.act_layer,
                    # layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 4, activation=None,
                        name = f"dense_d_2_w_{cur_width}"),#, kernel_regularizer=fc_reg)
                    # self.act_layer,
                    # layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 4, activation=None,
                        name = f"dense_d_3_w_{cur_width}"),#, kernel_regularizer=fc_reg)
                    # self.act_layer,
                    # layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 3, activation=None,
                        name = f"dense_d_4_w_{cur_width}"),#, kernel_regularizer=fc_reg)
                    # self.act_layer,
                    # layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim * 2, activation=None,
                        name = f"dense_d_5_w_{cur_width}"),#, kernel_regularizer=fc_reg)
                    # self.act_layer,
                    # layers.BatchNormalization(center = True),
                    layers.Dense(units=latent_dim, activation=None,
                        name = f"dense_d_6_w_{cur_width}"),#, kernel_regularizer=fc_reg)
                    # self.act_layer,
                    # layers.BatchNormalization(center = True),
                ]
            )
        self.mean_dense = layers.Dense(self.latent_dim, activation=None, name = "mean_dense") #kernel_regularizer = fc_reg)
        self.logvar_dense = layers.Dense(self.latent_dim, activation=None, name = "logvar_dense", #kernel_regularizer = fc_reg,
                                         kernel_initializer=tf.keras.initializers.Zeros())

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'width': len(self.blocks)
        })
        return config
    
    def build(self, input_shape):
        # No need to call super().build() as we'll be setting input_spec
        # This line isn't necessary if you don't need to explicitly manage input shapes
        input_shape = tf.TensorShape(input_shape)

        # Flatten layer doesn't need explicit build
        flattened_shape = [None, tf.reduce_prod(input_shape[1:]).numpy()]
        print(f"flattened shape: {flattened_shape}")

        # Force build the internal layers
        for block in self.blocks:
            for layer in block:
                layer.build(flattened_shape)
                flattened_shape = [flattened_shape[0], layer.units]

        # These layers need to be given an input shape
        self.mean_dense.build(flattened_shape)
        self.logvar_dense.build(flattened_shape)

        # Mark the model as built
        super().build(input_shape)


    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Use variable arguments to simplify reconstructing the object
    
    
    def call(self, x, training = False, return_activations = False):
        act_tracker = {}
        outputs = []
        # print(f"pre flatten x: {x.shape}")
        # x = self.drop(x, training = training)
        # for cur_conv in self.conv_layers:
        #     x = cur_conv(x, training = training)
        #     x = self.act_layer(x)
        #     x = self.pool_step(x)
        print(f"encoder x: {x.shape}")
        sub_x = layers.Flatten()(x) #self.drop(x, training = training)
        print(f"post flatten x: {sub_x.shape}")
        for block in self.blocks:

            for layer in block:
                # sub_x = self.drop(sub_x, training = training)
                sub_x = layer(sub_x, training = training)
                sub_x = self.act_layer(sub_x)
                act_tracker[layer.name] = tf.reduce_mean(tf.reshape(sub_x, [sub_x.shape[0], -1]), axis = 1)
            outputs.append(sub_x)

        y = sum(outputs)
        print("y encoder shape: ", y.shape)
        mean = self.mean_dense(y)
        print("mean shape: ", mean.shape)
        act_tracker[self.mean_dense.name] = tf.reduce_mean(tf.reshape(mean, [mean.shape[0], -1]), axis = 1)
        logvar = self.logvar_dense(y)
        act_tracker[self.logvar_dense.name] = tf.reduce_mean(tf.reshape(logvar, [logvar.shape[0], -1]), axis = 1)
        if return_activations:
            return mean, logvar, act_tracker
        return mean, logvar, {}