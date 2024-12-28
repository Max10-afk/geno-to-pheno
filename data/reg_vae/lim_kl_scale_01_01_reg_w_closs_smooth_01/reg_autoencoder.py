# Define the autoencoder model
@keras.saving.register_keras_serializable()
class autoencoder(Model):
    def __init__(self, latent_dim, encoder_width = 3, decoder_depth = 1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
    
    def get_config(self):
        config = super(autoencoder, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'encoder_width': self.encoder_width,
            'decoder_depth': self.decoder_depth
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def sample_z(self):
        eps = tf.random.normal(shape=self.mean.shape)
        return eps * tf.exp(self.logvar * .5) + self.mean
    
    @tf.function
    def call(self, x, training = False):
        child_x = x[1]
        parents_x = x[0]
        x = tf.concat([parents_x, tf.expand_dims(child_x, 1)], axis = 1)
        self.mean, self.logvar = self.encoder(x, training = training)
        self.sample = self.sample_z()
        y_logits = self.decoder(parents_x, self.sample, training = training)
        return y_logits