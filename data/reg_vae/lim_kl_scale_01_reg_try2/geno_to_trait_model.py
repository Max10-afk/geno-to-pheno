# Define the autoencoder model
l_relu = layers.LeakyReLU(negative_slope=0.5)
@keras.saving.register_keras_serializable()
class trait_pred(Model):
    def __init__(self, width = 5, depth = 5, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        self.depth = depth
        self.width = width

        for cur_sub_id in range(self.width):
            cur_sub_block = []
            
            for layer_id in range(self.depth, 1, -1):
                cur_sub_block.append(layers.Dense(units=(layer_id + 1) ** 4, kernel_regularizer=fc_reg))
                cur_sub_block[len(cur_sub_block)-1].name = f"dense_{layer_id}_w_{cur_sub_id}"
                cur_sub_block.append(l_relu)
            
            cur_sub_block.append(layers.Dense(units = 6, kernel_regularizer=fc_reg))
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
    
    def call(self, parent_phenos, embed_x, training = False, return_weights = False):
        outputs = []
        for cur_block in self.blocks:
            sub_x = embed_x
            for cur_layer in cur_block:
                sub_x = cur_layer(sub_x, training = training)
            outputs.append(sub_x)
        raw_weights = tf.math.reduce_sum(outputs, axis = 0)
        # Use dynamic reshaping
        batch_size = tf.shape(parent_phenos)[0]
        num_parent_phenos = tf.shape(parent_phenos)[1]
        weights = tf.reshape(raw_weights, (batch_size, num_parent_phenos, -1))
        
        scaling_weights = weights[..., :2]
        bias = weights[..., 2:]
        bias = tf.reshape(bias, (batch_size, -1))
        scaled_parents = scaling_weights * parent_phenos
        y = tf.reduce_sum(scaled_parents, axis=1) + bias

        if return_weights:
            return [y, scaling_weights, bias]
        return y