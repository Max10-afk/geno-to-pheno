# Define the autoencoder model
fc_reg = keras.regularizers.L2(1e-4)
l_relu = layers.LeakyReLU(negative_slope=0.5)
@keras.saving.register_keras_serializable()
class trait_pred(Model):
    def __init__(self, width = 5, depth = 5, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        self.depth = depth
        self.width = width
        self.sampling_layer = sampling_layer()

        for cur_sub_id in range(self.width):
            cur_sub_block = []
            
            for layer_id in range(self.depth, 1, -1):
                cur_sub_block.append(layers.Dense(units=(layer_id + 1) ** 4, activation=None))#, kernel_regularizer=fc_reg))#, kernel_regularizer=fc_reg))    
                cur_sub_block[len(cur_sub_block)-1].name = f"dense_{layer_id}_w_{cur_sub_id}"
                cur_sub_block.append(l_relu)
            
            cur_sub_block.append(layers.Dense(units = 6, activation=None, kernel_regularizer=fc_reg))
            cur_sub_block[len(cur_sub_block)-1].name = f"dense_final_w_{cur_sub_id}"
            cur_sub_block.append(l_relu)
            self.blocks.append(cur_sub_block)
        
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
    
    def call(self, x, return_weights = False):
        parent_phenos_pre = x[0]
        x_geno = x[1]
        x = self.sampling_layer(x_geno)
        outputs = []
        for cur_block in self.blocks:
            sub_x = x
            for cur_layer in cur_block:
                sub_x = cur_layer(sub_x)
            outputs.append(sub_x)
        raw_weights = tf.math.reduce_sum(outputs, axis = 0)
        # Use dynamic reshaping
        batch_size = tf.shape(parent_phenos_pre)[0]
        num_parent_phenos = tf.shape(parent_phenos_pre)[1]
        weights = tf.reshape(raw_weights, (batch_size, num_parent_phenos, -1))
        
        scaling_weights = weights[..., :2]
        bias = weights[..., 2:]
        bias = tf.reshape(bias, (batch_size, -1))
        scaled_parents = scaling_weights * parent_phenos_pre
        y = tf.reduce_sum(scaled_parents, axis=1) + bias

        if return_weights:
            return [y, scaling_weights, bias]
        return y