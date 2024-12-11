# Define the autoencoder model
@keras.saving.register_keras_serializable()
class reg_vae(Model):
    def __init__(self, latent_dim, encoder_width = 3, decoder_depth = 1, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
        self.regressor = trait_pred()
        self.reg_vae_opt = optimizers.AdamW(learning_rate=1e-3)
        self.vae_opt = optimizers.AdamW(learning_rate=1e-3)
        self.loss_fn = elbo_loss()
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.kl_scale_tracker = tf.keras.metrics.Mean(name="kl_scale")
        self.elbo_rec_tracker = tf.keras.metrics.Mean(name="elbo_rec")
        self.elbo_reg_tracker = tf.keras.metrics.Mean(name="elbo_reg")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        self.mean_loss_tracker = tf.keras.metrics.MeanAbsolutePercentageError(name = "mean_deviation")
        self.sd_loss_tracker = tf.keras.metrics.MeanAbsolutePercentageError(name = "sd_deviation")
        self.epoch_tracker = tf.keras.metrics.Mean(name="epoch")
        self.cat_acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="cat_acc")
        self.class_acc_tracker = [tf.keras.metrics.Accuracy(name=f"{cur_class}_acc") for cur_class in range(11)]
    
    def get_config(self):
        config = super(reg_vae, self).get_config()
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
    def call(self, data, training = False):
        geno_x, trait_x = data
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[0]
        child_genos = geno_x[1]
        c_geno_x = tf.concat([geno_x[0], tf.expand_dims(geno_x[1], 1)], axis = 1)
        mean, logvar = self.encoder(c_geno_x, training = training)
        embed_x = self.sample_z(mean, logvar)
        geno_logits = self.decoder(parents_genos, embed_x, training = training)
        pheno_pred = self.regressor(parent_trait, embed_x, training = training)

        return embed_x, geno_logits, pheno_pred

    @tf.function
    def update_trackers(self, elbo_rec_loss, elbo_reg_loss, reconstruction_loss, kl_loss, kl_scale, reg_loss,
                        geno_labels, geno_logits, cur_epoch, trait_pred, trait_true):
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.elbo_rec_tracker.update_state(elbo_rec_loss)
        self.elbo_reg_tracker.update_state(elbo_reg_loss)
        self.cat_acc_tracker.update_state(geno_labels, geno_logits)
        self.kl_scale_tracker.update_state(kl_scale)
        self.epoch_tracker.update_state(cur_epoch)
        self.reg_loss_tracker.update_state(reg_loss)
        self.mean_loss_tracker.update_state(trait_true[:, 0], trait_pred[:, 0])
        self.sd_loss_tracker.update_state(trait_true[:, 1], trait_pred[:, 1])

        for class_id in range(len(self.class_acc_tracker)):
            class_mask = tf.equal(tf.argmax(geno_labels, axis=-1), class_id)
            class_y_truth = tf.boolean_mask(tf.argmax(geno_labels, axis=-1), class_mask)
            class_y_pred = tf.boolean_mask(tf.argmax(geno_logits, axis=-1), class_mask)
            self.class_acc_tracker[class_id].update_state(class_y_truth, class_y_pred)
        
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step(self, data, cur_epoch):
        training = True
        geno_x, trait_x = data
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[0]
        child_genos = geno_x[1]

        with tf.GradientTape(persistent = True) as grad_tape:
            c_geno_x = tf.concat([geno_x[0], tf.expand_dims(geno_x[1], 1)], axis = 1)
            mean, logvar = self.encoder(c_geno_x, training = training)
            embed_x = self.sample_z(mean, logvar)
            geno_logits = self.decoder(parents_genos, embed_x, training = training)
            pheno_pred = self.regressor(parent_trait, embed_x, training = training)
            elbo_reg_loss, elbo_rec_loss, kl_loss, kl_scale, reg_loss, rec_loss, trait_pred, trait_true  = self.loss_fn(child_genos, geno_logits, mean, logvar,
                trait_pred = pheno_pred, trait_true = child_trait, epoch = cur_epoch)
        vae_grads = grad_tape.gradient(elbo_rec_loss, self.encoder.trainable_variables +\
            self.decoder.trainable_variables)
        self.vae_opt.apply_gradients(zip(vae_grads, self.encoder.trainable_variables +\
            self.decoder.trainable_variables))
        trait_grads = grad_tape.gradient(elbo_reg_loss, self.encoder.trainable_variables +\
            self.regressor.trainable_variables)
        self.reg_vae_opt.apply_gradients(zip(trait_grads, self.encoder.trainable_variables +\
            self.regressor.trainable_variables))
        # enc_grads = grad_tape.gradient(kl_loss * tf.cast(kl_scale, tf.float32) + reg_loss + rec_loss, self.encoder.trainable_variables +\
        #     self.decoder.trainable_variables + self.regressor.trainable_variables)
        # self.enc_opt.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        # dec_grads = grad_tape.gradient(rec_loss, self.decoder.trainable_variables)
        # self.dec_opt.apply_gradients(zip(dec_grads, self.decoder.trainable_variables))
        # reg_grads = grad_tape.gradient(reg_loss, self.regressor.trainable_variables)
        # self.reg_opt.apply_gradients(zip(reg_grads, self.regressor.trainable_variables))
        del grad_tape
        return self.update_trackers(elbo_rec_loss, elbo_reg_loss, rec_loss, kl_loss, kl_scale, reg_loss,
                                    child_genos, geno_logits, cur_epoch, trait_pred, child_trait)

    @tf.function
    def test_step(self, data, cur_epoch):
        training = False
        geno_x, trait_x = data
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[0]
        child_genos = geno_x[1]


        geno_x = tf.concat([geno_x[0], tf.expand_dims(geno_x[1], 1)], axis = 1)
        mean, logvar = self.encoder(geno_x, training = training)
        embed_x = self.sample_z(mean, logvar)
        geno_logits = self.decoder(parents_genos, embed_x, training = training)
        pheno_pred = self.regressor(parent_trait, embed_x, training = training)
        # def call(self, x_labels, x_logits, mean, logvar, trait_pred, trait_true, epoch = None, step_size = 50):
        elbo_reg_loss, elbo_rec_loss, kl_loss, kl_scale, reg_loss, rec_loss, trait_pred, trait_true  = self.loss_fn(child_genos, geno_logits, mean, logvar,
            trait_pred = pheno_pred, trait_true = child_trait, epoch = cur_epoch)

        return self.update_trackers(elbo_rec_loss, elbo_reg_loss, rec_loss, kl_loss, kl_scale, reg_loss,
                            child_genos, geno_logits, cur_epoch, trait_pred, trait_true)




    @property
    def metrics(self):
        return [self.kl_loss_tracker, self.kl_scale_tracker, self.elbo_rec_tracker,
                self.elbo_reg_tracker, self.reg_loss_tracker, self.epoch_tracker, self.cat_acc_tracker,
                self.reconstruction_loss_tracker, self.mean_loss_tracker, self.sd_loss_tracker] + self.class_acc_tracker