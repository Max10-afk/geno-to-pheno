class encoder(Model):
    def __init__(self, latent_dim, width=1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.mean_blocks = []
        self.act_layer = ACT_LAYER
        num_conv_iterations = 3
        self.conv_geno_layers = []
        self.embedding = layers.Embedding(11, 5, name="ini_embedding")
        self.fc_reg = tf.keras.regularizers.L2(l2=0.03)

        for i in range(num_conv_iterations):
            filter_size = 2 * (i + 2)
            cur_conv_channel = "channels_last"  #  if i % 2 == 0 else "channels_first"
            self.conv_geno_layers.append(
                tf.keras.layers.Conv1D(
                    filter_size,
                    kernel_size=4,
                    activation=self.act_layer,
                    name=f"geno_conv_{i}",
                    data_format=cur_conv_channel,
                )
            )
            self.conv_geno_layers.append(
                layers.AveragePooling1D(
                    pool_size=2, data_format=cur_conv_channel, name=f"geno_maxpool_{i}"
                )
            )
        self.conv_diff_layers = []
        self.embedding = layers.Embedding(11, 8, name="ini_embedding")

        for i in range(num_conv_iterations):
            filter_size = 2 * (i + 2)
            cur_conv_channel = "channels_last"  #  if i % 2 == 0 else "channels_first"
            self.conv_diff_layers.append(
                tf.keras.layers.Conv1D(
                    filter_size,
                    kernel_size=4,
                    activation=self.act_layer,
                    name=f"diff_conv_{i}",
                    data_format=cur_conv_channel,
                )
            )
            self.conv_diff_layers.append(
                layers.AveragePooling1D(
                    pool_size=2, data_format=cur_conv_channel, name=f"diff_maxpool_{i}"
                )
            )
        # self.post_conv_dense = layers.Dense(units = 300, activation = self.act_layer)
        self.p_to_c_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=8,  # Match embedding dim
            value_dim=8,
            dropout=0.0
        )
        self.d_to_c_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=8,  # Match embedding dim
            value_dim=8,
            dropout=0.0
        )

        b_size = 16
        seq_len = 1144
        block_size = 143
        att_mask = tf.ones([b_size, seq_len, seq_len])
        band_mask = tf.ones(
            [b_size, 1, seq_len // block_size - 4, block_size, 3 * block_size]
        )
        from_mask = tf.ones([b_size, 1, seq_len, 1])
        to_mask = tf.ones([b_size, 1, 1, seq_len])
        from_blocked_mask = tf.ones([b_size, seq_len // block_size, block_size])
        to_blocked_mask = tf.ones([b_size, seq_len // block_size, block_size])
        self.masks = [
            att_mask,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
        ]
        # self.ini_p_to_p_attention = MultiHeadedAttentionLayer(
        #     attention_type="block_sparse",
        #     num_attention_heads=1,
        #     name="test",
        #     size_per_head=5,
        #     from_seq_length=1144,
        #     to_seq_length=1144,
        #     from_block_size=block_size,
        #     to_block_size=block_size,
        # )
        self.ini_p_to_p_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=8,  # Match embedding dim
            value_dim=8,
            dropout=0.3
        )

        self.diff_to_c_attention = tf.keras.layers.Attention(
            use_scale=False, score_mode="dot", dropout=0.3, seed=None, **kwargs
        )
        self.diff_to_c_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=8,  # Match embedding dim
            value_dim=8,
            dropout=0.3
        )
        self.logvar_block = []
        for cur_width in range(width):
            self.logvar_block.append(
                [
                    layers.Dense(
                        units=latent_dim * 4,
                        activation=None,
                        name=f"encoder_dense_d_3_w_{cur_width}",
                    ),
                    layers.Dense(
                        units=latent_dim * 3,
                        activation=None,
                        name=f"encoder_dense_d_4_w_{cur_width}",
                    ),
                    layers.Dense(
                        units=latent_dim * 2,
                        activation=None,
                        name=f"encoder_dense_d_5_w_{cur_width}",
                    ),
                    layers.Dense(
                        units=latent_dim,
                        activation=None,
                        name=f"encoder_dense_d_6_w_{cur_width}",
                    ),
                ]
            )

        self.drop = tf.keras.layers.Dropout(0.4)
        for cur_width in range(width):
            self.mean_blocks.append(
                [
                    # layers.Dense(
                    #     units=latent_dim * 4,
                    #     activation=None,
                    #     name=f"dense_d_3_w_{cur_width}",
                    #     kernel_regularizer=self.fc_reg,
                    # ),
                    # layers.Dense(
                    #     units=latent_dim * 3,
                    #     activation=None,
                    #     name=f"dense_d_4_w_{cur_width}",
                    #     kernel_regularizer=self.fc_reg,
                    # ),
                    layers.Dense(
                        units=latent_dim * 2,
                        activation=None,
                        name=f"dense_d_5_w_{cur_width}",
                        kernel_regularizer=self.fc_reg,
                    ),
                    layers.Dense(
                        units=latent_dim,
                        activation=None,
                        name=f"dense_d_6_w_{cur_width}",
                        kernel_regularizer=self.fc_reg,
                    ),
                ]
            )
        self.mean_dense = layers.Dense(
            self.latent_dim, activation=None, name="mean_dense"
        )  # kernel_regularizer = fc_reg)
        self.logvar_dense = layers.Dense(
            self.latent_dim,
            activation=None,
            name="logvar_dense",  # kernel_regularizer = fc_reg,
            kernel_initializer=tf.keras.initializers.Zeros(),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim, "width": len(self.mean_blocks)})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            **config
        )  # Use variable arguments to simplify reconstructing the object

    def call(self, p_genos, c_geno=None, training=False, return_activations=False):
        act_tracker = {}
        geno_x = self.embedding(p_genos, training=training)
        # print("post embedding geno_x.shape: ", geno_x.shape)
        act_tracker[self.embedding.name] = tf.reduce_mean(
            tf.reshape(geno_x, [geno_x.shape[0], -1]), axis=1
        )
        geno_x = tf.split(geno_x, num_or_size_splits=geno_x.shape[1], axis=1)
        pos_info = positional_encoding(geno_x[0].shape[2], geno_x[0].shape[3])
        geno_x = [tf.squeeze(cur_geno) for cur_geno in geno_x] # + pos_info
        diff_in = []
        diff_x = tf.squeeze((geno_x[0] - geno_x[1]) * geno_x[2])
        diff_in.append(diff_x)
        att_geno_1 = self.ini_p_to_p_attention(
            tf.squeeze(geno_x[0]),
            tf.squeeze(geno_x[2]),
            training=training,
        )
        geno_x.append(att_geno_1)
        
        c_scaled_by_d = self.d_to_c_attention(
            diff_x,
            tf.squeeze(geno_x[2]),
            training=training
        )
        diff_in.append(c_scaled_by_d)
        
        
        # geno_x = [tf.squeeze(cur_geno) for cur_geno in geno_x]
        geno_out = []
        for cur_geno in geno_x:
            cur_geno = tf.squeeze(cur_geno)
            for cur_conv in self.conv_geno_layers:
                cur_geno = cur_conv(cur_geno, training=training)
                print("encoder conv shape: ", cur_geno.shape)
                act_tracker["mean_" + cur_conv.name] = tf.reduce_mean(
                    tf.reshape(cur_geno, [cur_geno.shape[0], -1]), axis=1
                )
            geno_out.append(cur_geno)
        diff_out = []
        for cur_diff in diff_in:
          for cur_conv in self.conv_diff_layers:
              cur_diff = cur_conv(cur_diff, training=training)
              print("encoder conv shape: ", cur_diff.shape)
              act_tracker["mean_" + cur_conv.name] = tf.reduce_mean(
                  tf.reshape(cur_diff, [cur_diff.shape[0], -1]), axis=1
              )
          diff_out.append(cur_diff)

        # geno_x = [layers.Flatten()(cur_geno) for cur_geno in geno_out]
        print("encoder post conv shape: ", geno_out[0].shape)
        # c_scaled_by_p = self.p_to_c_attention(
        #     geno_out[0], geno_out[2], training=training
        # )
        # c_scaled_by_d = self.d_to_c_attention(
        #     diff_x, geno_out[2], training=training
        # )
        # c_scaled_by_diff = self.diff_to_c_attention([diff_out[0], geno_out[2]], training = training)
        # print("encoder scaled_c shape: ", c_scaled_by_p.shape)
        geno_x = tf.concat(
            [
                layers.Flatten()(geno_out[2]),
                layers.Flatten()(geno_out[3]),
                layers.Flatten()(diff_out[0]),
                layers.Flatten()(diff_out[1])
            ],
            axis=1,
        )

        act_tracker["post_conv_concat"] = tf.reduce_mean(
            tf.reshape(geno_x, [geno_x.shape[0], -1]), axis=1
        )
        mean_outputs = []
        for block in self.mean_blocks:
            sub_x = self.drop(geno_x, training=training)
            for layer in block:
                sub_x = layer(sub_x, training=training)
                sub_x = self.act_layer(sub_x)
                act_tracker["mean_" + layer.name] = tf.reduce_mean(
                    tf.reshape(sub_x, [sub_x.shape[0], -1]), axis=1
                )
            mean_outputs.append(sub_x)
        # logvar_outputs = []
        # for block in self.logvar_block:
        #     sub_x = geno_x
        #     for layer in block:
        #         sub_x = layer(sub_x, training = training)
        #         sub_x = self.act_layer(sub_x)
        #         act_tracker["logvar_" + layer.name] = tf.reduce_mean(tf.reshape(sub_x, [sub_x.shape[0], -1]), axis = 1)
        #     logvar_outputs.append(sub_x)

        mean = tf.concat(mean_outputs, axis=1)
        mean = self.mean_dense(mean, training=training)
        act_tracker["mean_" + self.mean_dense.name] = tf.reduce_mean(
            tf.reshape(mean, [mean.shape[0], -1]), axis=1
        )
        logvar = tf.concat(mean_outputs, axis=1)
        logvar = self.logvar_dense(logvar, training=training)
        if return_activations:
            return mean, logvar, act_tracker
        return mean, logvar, {}
