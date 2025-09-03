# Define the autoencoder model
@keras.saving.register_keras_serializable()
class trait_pred(Model):
    def __init__(self, width = 5, depth = 5, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        self.depth = depth
        self.width = width
        # self.drop = feature_drop_layer(0.1, feature_dim = 1)
        self.drop = tf.keras.layers.Dropout(0.6)
        num_conv_iterations = 5
        self.conv_layers = []
        conv_kernal_size = 2
        self.act_layer = tf.keras.layers.LeakyReLU(alpha = 0.1)
        self.pool_step = tf.keras.layers.MaxPooling1D(pool_size=2,
            data_format = "channels_first")
        for i in range(num_conv_iterations):
            filter_size = 20 * (i + 1)
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    filter_size,
                    kernel_size=conv_kernal_size,
                    activation=None,
                    name="CONV_" + str(i),
                    data_format = "channels_first"
                )
            )
            self.conv_layers[-1].name = f"conv_{i}"

        for cur_width in range(self.width):
            cur_sub_block = []
            
            for layer_id in range(self.depth, 1, -1):
                cur_sub_block.append(self.drop)
                cur_sub_block.append(layers.Dense(units=(layer_id + 1) ** 4, kernel_regularizer=fc_reg))
                cur_sub_block[len(cur_sub_block)-1].name = f"dense_d_{layer_id}_w_{cur_width}"
                cur_sub_block.append(self.act_layer)
            
            cur_sub_block.append(layers.Dense(units = 6, kernel_regularizer=fc_reg))
            cur_sub_block[len(cur_sub_block)-1].name = f"dense_final_w_{cur_width}"
            cur_sub_block.append(self.act_layer)
            self.blocks.append(cur_sub_block)
        self.sd_layer = layers.Dense(units = 6)#, kernel_regularizer=fc_reg
        self.sd_layer1 = layers.Dense(units = 3)#, kernel_regularizer=fc_reg
        self.mean_layer = layers.Dense(units = 6)#, kernel_regularizer=fc_reg
        self.mean_layer1 = layers.Dense(units = 3)#, kernel_regularizer=fc_reg
        self.manual_act = self.act_layer
        
    def get_config(self):
        config = super(trait_pred, self).get_config()
        config.update({
            'width': self.width,
            'depth': self.depth
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Use variable arguments to simplify reconstructing the object
    
    
    def call(self, parent_phenos, embed_x, training = False, return_weights = False,
        return_activations = False):
        act_tracker = {}
        embed_x = tf.expand_dims(embed_x, axis = 1)
        for cur_conv in self.conv_layers:
            embed_x = cur_conv(embed_x, training = training)
            embed_x = self.act_layer(embed_x)
            embed_x = self.pool_step(embed_x)
            act_tracker[cur_conv.name] = tf.reduce_mean(tf.reshape(embed_x, [embed_x.shape[0], -1]), axis = 1)
        embed_x = self.pool_step(embed_x)
        embed_x = tf.squeeze(embed_x)
        outputs = []
        # cur_sub_block.append(self.drop)
        # cur_sub_block.append(layers.Dense(units=(layer_id + 1) ** 4, kernel_regularizer=fc_reg))
        # cur_sub_block[len(cur_sub_block)-1].name = f"dense_{layer_id}_w_{cur_width}"
        # cur_sub_block.append(self.act_layer)
        for cur_block in self.blocks:
            sub_x = embed_x
            for cur_layer in cur_block:
                sub_x = self.drop(sub_x, training = training)
                sub_x = cur_layer(sub_x, training = training)
                sub_x = self.act_layer(sub_x)
                act_tracker[cur_layer.name] = tf.reduce_mean(tf.reshape(sub_x, [sub_x.shape[0], -1]), axis = 1)
            outputs.append(sub_x)
        block_out = tf.math.reduce_sum(outputs, axis = 0)
        act_tracker["block_out"] = tf.reduce_mean(tf.reshape(sub_x, [sub_x.shape[0], -1]), axis = 1)
        sd_weights = self.sd_layer(block_out)
        sd_weights = self.manual_act(sd_weights)
        act_tracker["sd_weights"] = tf.reduce_mean(tf.reshape(sd_weights, [sd_weights.shape[0], -1]), axis = 1)
        sd_weights = self.sd_layer1(sd_weights)
        sd_weights = self.manual_act(sd_weights)
        act_tracker["sd_weights_1"] = tf.reduce_mean(tf.reshape(sd_weights, [sd_weights.shape[0], -1]), axis = 1)
        mean_weights = self.mean_layer(block_out)
        mean_weights = self.manual_act(mean_weights)
        act_tracker["mean_weights"] = tf.reduce_mean(tf.reshape(mean_weights, [mean_weights.shape[0], -1]), axis = 1)
        mean_weights = self.mean_layer1(mean_weights)
        mean_weights = self.manual_act(mean_weights)
        act_tracker["mean_weights_1"] = tf.reduce_mean(tf.reshape(mean_weights, [mean_weights.shape[0], -1]), axis = 1)
        
        raw_weights = tf.concat([mean_weights, sd_weights], axis = 1)
        # Use dynamic reshaping
        batch_size = tf.shape(parent_phenos)[0]
        num_parent_phenos = tf.shape(parent_phenos)[1]
        weights = tf.reshape(raw_weights, (batch_size, num_parent_phenos, -1))
        
        scaling_weights = weights[..., :2]
        bias = weights[..., 2:]
        bias = tf.reshape(bias, (batch_size, -1))
        scaled_parents = scaling_weights * parent_phenos
        y = tf.reduce_sum(scaling_weights, axis=1) + bias

        if return_weights:
            return y, scaling_weights, bias
        if return_activations:
            return y, act_tracker
        return y, {}