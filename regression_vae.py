# Define the autoencoder model
@keras.saving.register_keras_serializable()
class regression_vae(Model):
    def __init__(self, latent_dim, encoder_width = 3, decoder_depth = 1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.current_step = tf.Variable(initial_value = 1., trainable = False)

        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
        self.regressor = trait_pred()
        self.loss_fn = elbo_loss()
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.cat_acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="cat_acc")
        self.class_acc_tracker = [tf.keras.metrics.CategoricalAccuracy(name=f"{cur_class}_acc") for cur_class in range(11)]
    
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

    def sample_z(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def call(self, x, training = False):
        child_x = x[:, 0, ...]
        parents_x = x[:, 1:3, ...]

        mean, logvar = self.encoder(x, training = training)
        embed_x = self.sample_z(mean, logvar)
        y_logits = self.decoder(parents_x, embed_x, training = training)
        return y_logits
    
    def train_step(self, data):
        self.current_step.assign_add(1.)
        training = True
        x, y = data
        child_x = x[:, 0, ...]
        parents_x = x[:, 1:3, ...]

        with tf.GradientTape() as tape:
            mean, logvar = self.encoder(x, training = training)
            embed_x = self.sample_z(mean, logvar)
            y_logits = self.decoder(parents_x, embed_x, training = training)
            elbo, reconstruction_loss, kl_loss = self.loss_fn(child_x, y_logits, mean, logvar,
                epoch = self.current_step.value(), no_kl = self.climbing)
        
        grads = tape.gradient(elbo, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Track metrics
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.elbo_tracker.update_state(elbo)
        self.cat_acc_tracker.update_state(y, y_logits)

        # Class-wise accuracy update
        for class_id in range(len(self.class_acc_tracker)):
            class_mask = tf.equal(tf.argmax(y_logits, axis=-1), class_id)
            class_y = tf.boolean_mask(y, class_mask)
            class_y_logits = tf.boolean_mask(y_logits, class_mask)
            self.class_acc_tracker[class_id].update_state(class_y, class_y_logits)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        training = False
        # Unpack the data
        x, y = data
        child_x = x[:, 0, ...]
        parents_x = x[:, 1:3, ...]

        # Compute predictions
        mean, logvar = self.encoder(x, training = training)
        embed_x = self.sample_z(mean, logvar)
        y_logits = self.decoder(parents_x, embed_x, training = training)

        elbo, reconstruction_loss, kl_loss = self.loss_fn(child_x, y_logits, mean, logvar)

        # Updates the metrics tracking the loss
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.elbo_tracker.update_state(elbo)
        self.cat_acc_tracker.update_state(y, y_logits)
        self.trainable_vars_tracker.update_state(len(self.trainable_variables))

        for class_id in range(len(self.class_acc_tracker)):
            class_mask = tf.equal(tf.argmax(y_logits, axis=-1), class_id)
            class_y = tf.boolean_mask(y, class_mask)
            class_y_logits = tf.boolean_mask(y_logits, class_mask)
            self.class_acc_tracker[class_id].update_state(class_y, class_y_logits)
        return {m.name: m.result() for m in self.metrics}
    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.kl_loss_tracker, self.elbo_tracker,
            self.cat_acc_tracker] + self.class_acc_tracker 