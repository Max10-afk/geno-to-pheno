# Define the autoencoder model
@keras.saving.register_keras_serializable()
class autoencoder(Model):
    def __init__(self, latent_dim, encoder_width = 3, decoder_depth = 1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.current_step = tf.Variable(initial_value = 1., trainable = False)
        self.old_mi = tf.Variable(initial_value = -np.inf, trainable = False)
        self.climbing = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)
        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
        self.loss_fn = elbo_loss()
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.climbing_tracker = tf.keras.metrics.Mean(name="climbing")
        self.trainable_vars_tracker = tf.keras.metrics.Mean(name="trainable_vars_tracker")
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

    def log_sum_exp(self, value, axis):
        m = tf.math.reduce_max(value, axis=axis, keepdims=True)
        return tf.math.reduce_logsumexp(value - m, axis=axis, keepdims=False) + tf.squeeze(m, axis=axis)
    
    def aggressiveness_switch(self, dec_bool):
        to_bool = tf.cond(
    	    pred = dec_bool,
    	    true_fn = lambda: False,
    	    false_fn = lambda: True
    	)
        for layer in self.decoder.layers:
            # print(f"prev {layer}.trainable: {layer.trainable}")
            layer.trainable = to_bool
            # print(f"updated {layer}.trainable to {layer.trainable}")
        

    @tf.function
    def mutual_info(self, x):
        # Compute mutual information content between embedding and
        # input for lagging inference VAE, see https://arxiv.org/abs/1901.05534v2

        # Forward pass to calculate mean and log variance
        mu, logvar = self.encoder(x)
        x_batch, nz = mu.shape
        # Negative entropy term
        neg_entropy = tf.math.reduce_mean(-0.5 * nz * tf.math.log(2 * np.pi) - 0.5 * tf.reduce_sum(1 + logvar, axis=-1))
        z_samples = self.sample_z(mu, logvar)

        # Add Null dimension for correct broadcasting
        mu = tf.expand_dims(mu, axis=0)
        logvar = tf.expand_dims(logvar, axis=0)
        var = tf.math.exp(logvar)

        # Aggregate posterior
        dev = z_samples - mu
        log_density = -0.5 * (tf.math.reduce_sum((dev ** 2) / var, axis=-1)) - \
                      0.5 * (nz * tf.math.log(2 * np.pi) + tf.math.reduce_sum(logvar, axis=-1))
        log_qz = self.log_sum_exp(log_density, axis=1) - np.log(x_batch)
        return (neg_entropy - tf.math.reduce_mean(log_qz, axis=-1))
    
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
        new_mi = self.mutual_info(x)
        climbing_cond = tf.reduce_any(new_mi > self.old_mi)
        self.climbing.assign(climbing_cond)
        self.old_mi.assign(new_mi)

        elbo, reconstruction_loss, kl_loss = self.loss_fn(child_x, y_logits, mean, logvar,
            no_kl = self.climbing)

        # Part of lagging inference VAE, freezes decoder weights
        # when climbing == False
        # self.aggressiveness_switch(self.climbing)

        # Updates the metrics tracking the loss
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.elbo_tracker.update_state(elbo)
        self.cat_acc_tracker.update_state(y, y_logits)
        self.climbing_tracker.update_state(self.climbing)
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
            self.cat_acc_tracker, self.climbing_tracker, self.trainable_vars_tracker] + self.class_acc_tracker 