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
        self.kl_scale_tracker = tf.keras.metrics.Mean(name="kl_scale")
        self.lr_tracker = tf.keras.metrics.Mean(name="lr")
        self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.epoch_tracker = tf.keras.metrics.Mean(name="epoch")
        self.cat_acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="cat_acc")
        self.class_acc_tracker = [tf.keras.metrics.Accuracy(name=f"{cur_class}_acc") for cur_class in range(11)]
    
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
    
    @tf.function
    def call(self, x, training = False):
        child_x = x[:, 0, ...]
        parents_x = x[:, 1:3, ...]

        mean, logvar = self.encoder(x, training = training)
        embed_x = self.sample_z(mean, logvar)
        y_logits = self.decoder(parents_x, embed_x, training = training)
        return y_logits

    @tf.function
    def train_step(self, data, cur_epoch):
        self.current_step.assign_add(1.)
        training = True
        x, y = data
        child_x = x[:, 0, ...]
        parents_x = x[:, 1:3, ...]

        with tf.GradientTape() as tape:
            mean, logvar = self.encoder(x, training = training)
            embed_x = self.sample_z(mean, logvar)
            y_logits = self.decoder(parents_x, embed_x, training = training)
            elbo, reconstruction_loss, kl_loss, kl_scale = self.loss_fn(child_x, y_logits, mean, logvar,
                epoch = cur_epoch, no_kl = self.climbing)
        
        grads = tape.gradient(elbo, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Track metrics
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.elbo_tracker.update_state(elbo)
        self.cat_acc_tracker.update_state(y, y_logits)
        self.kl_scale_tracker.update_state(kl_scale)
        self.epoch_tracker.update_state(cur_epoch)
        self.lr_tracker.update_state(self.optimizer.learning_rate)

        for class_id in range(len(self.class_acc_tracker)):
            class_mask = tf.equal(tf.argmax(child_x, axis=-1), class_id)
            class_y_truth = tf.boolean_mask(tf.argmax(child_x, axis=-1), class_mask)
            class_y_pred = tf.boolean_mask(tf.argmax(y_logits, axis=-1), class_mask)
            self.class_acc_tracker[class_id].update_state(class_y_truth, class_y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data, cur_epoch):
        training = False
        # Unpack the data
        x, y = data
        child_x = x[:, 0, ...]
        parents_x = x[:, 1:3, ...]

        # Compute predictions
        mean, logvar = self.encoder(x, training = training)
        embed_x = self.sample_z(mean, logvar)
        y_logits = self.decoder(parents_x, embed_x, training = training)
        

        elbo, reconstruction_loss, kl_loss, kl_scale = self.loss_fn(child_x, y_logits, mean, logvar,
            no_kl = self.climbing, epoch = cur_epoch)

        # Part of lagging inference VAE, freezes decoder weights
        # when climbing == False
        # self.aggressiveness_switch(self.climbing)

        # Updates the metrics tracking the loss
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.elbo_tracker.update_state(elbo)
        self.cat_acc_tracker.update_state(y, y_logits)
        self.kl_scale_tracker.update_state(kl_scale)
        self.epoch_tracker.update_state(cur_epoch)
        self.lr_tracker.update_state(self.optimizer.learning_rate)

        for class_id in range(len(self.class_acc_tracker)):
            class_mask = tf.equal(tf.argmax(child_x, axis=-1), class_id)
            class_y_truth = tf.boolean_mask(tf.argmax(child_x, axis=-1), class_mask)
            class_y_pred = tf.boolean_mask(tf.argmax(y_logits, axis=-1), class_mask)
            self.class_acc_tracker[class_id].update_state(class_y_truth, class_y_pred)

        return {m.name: m.result() for m in self.metrics}


    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.kl_loss_tracker, self.elbo_tracker,
            self.cat_acc_tracker, self.epoch_tracker, self.kl_scale_tracker, self.lr_tracker] + self.class_acc_tracker