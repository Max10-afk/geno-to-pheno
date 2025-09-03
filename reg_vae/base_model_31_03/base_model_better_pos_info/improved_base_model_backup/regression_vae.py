# Define the autoencoder model
@tf.keras.utils.register_keras_serializable()
class reg_vae(Model):
    def __init__(
        self, latent_dim, encoder_width=3, decoder_depth=1, pheno_only=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
        self.regressor = trait_pred(width=5, depth=1)
        self.pheno_only = pheno_only
        self.loss_fn = elbo_loss()
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.kl_scale_tracker = tf.keras.metrics.Mean(name="kl_scale")
        self.elbo_rec_tracker = tf.keras.metrics.Mean(name="elbo_rec")
        self.elbo_reg_tracker = tf.keras.metrics.Mean(name="elbo_reg")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        self.mean_loss_tracker = tf.keras.metrics.MeanAbsolutePercentageError(
            name="mean_deviation"
        )
        self.sd_loss_tracker = tf.keras.metrics.MeanAbsolutePercentageError(
            name="sd_deviation"
        )
        self.epoch_tracker = tf.keras.metrics.Mean(name="epoch")
        self.cat_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="cat_acc"
        )
        self.class_acc_tracker = [
            tf.keras.metrics.SparseCategoricalAccuracy(name=f"{cur_class}_acc")
            for cur_class in range(11)
        ]
        self.reg_opt = optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)
        self.dec_opt = optimizers.AdamW(learning_rate=1e-3)
        self.enc_opt = optimizers.AdamW(learning_rate=1e-4)
        self.prev_reg_loss = tf.Variable(initial_value=1.0, trainable=False)
        self.prev_rec_loss = tf.Variable(initial_value=1.0, trainable=False)
        self.train_cycle_count = tf.Variable(initial_value=0.0, trainable=False)
        self.single_opt_cycle_count = tf.Variable(initial_value=0.0, trainable=False)
        self.all_opt_cycle_count = tf.Variable(initial_value=0.0, trainable=False)

    def get_config(self):
        config = super(reg_vae, self).get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "encoder_width": self.encoder_width,
                "decoder_depth": self.decoder_depth,
                "pheno_only": self.pheno_only,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def sample_z(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    @tf.function
    def call(self, data, training=False, return_activations=False):

        geno_x, trait_x, meta_x = data
        seq_pos, chr_pos, pop_x = meta_x
        parent_trait = trait_x
        parents_genos = geno_x[:, :2, ...]
        mean, logvar, enc_act = self.encoder.call(
            geno_x, meta_x, training=training, return_activations=return_activations
        )
        embed_x = self.sample_z(mean, logvar)
        geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
            parents_genos,
            embed_x,
            training=training,
            return_activations=return_activations,
        )
        pheno_pred, reg_act, reg_gate = self.regressor.call(
            parent_trait,
            embed_x,
            parents_genos,
            training=training,
            return_activations=return_activations,
        )
        all_activations = {"encoder": enc_act, "decoder": dec_act, "regressor": reg_act}
        if self.pheno_only:
            return pheno_pred
        return (
            embed_x,
            geno_logits,
            pheno_pred,
            mean,
            logvar,
            dec_gate,
            reg_gate,
            all_activations,
            geno_pred,
        )

    @tf.function
    def update_trackers(
        self,
        elbo_rec_loss,
        elbo_reg_loss,
        reconstruction_loss,
        kl_loss,
        kl_scale,
        reg_loss,
        geno_labels,
        geno_logits,
        cur_epoch,
        trait_pred,
        trait_true,
    ):
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
            class_mask = tf.equal(geno_labels, class_id)
            class_y_truth = tf.boolean_mask(geno_labels, class_mask)
            class_y_pred = tf.boolean_mask(
                tf.nn.softmax(geno_logits, axis=-1), class_mask
            )
            self.class_acc_tracker[class_id].update_state(class_y_truth, class_y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step(self, data, cur_epoch, return_activations=False):
        training = True
        geno_x, trait_x, meta_x = data
        seq_pos, chr_pos, pop_x = meta_x
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[:, :2, ...]
        child_genos = geno_x[:, 2, ...]
        kl_scale = 0.01
        with tf.GradientTape(persistent=True) as grad_tape:
            mean, logvar, enc_act = self.encoder.call(
                geno_x, meta_x, training=training, return_activations=return_activations
            )
            logvar = tf.clip_by_value(logvar, -7, 7)
            mean = tf.clip_by_value(mean, -30, 30)
            embed_x = self.sample_z(mean, logvar)
            geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
                parents_genos,
                embed_x,
                training=training,
                return_activations=return_activations,
            )
            pheno_pred, reg_act, reg_gate = self.regressor.call(
                parent_trait,
                embed_x,
                parents_genos,
                training=training,
                return_activations=return_activations,
            )
            kl_loss, reg_loss, rec_loss = self.loss_fn(
                child_genos,
                geno_logits,
                mean,
                logvar,
                trait_pred=pheno_pred,
                trait_true=child_trait,
                epoch=cur_epoch,
            )
            elbo_rec_loss = rec_loss
            elbo_reg_loss = reg_loss
            total_loss = kl_loss * kl_scale + reg_loss + rec_loss * 1000
        self.prev_reg_loss.assign(reg_loss)
        self.prev_rec_loss.assign(rec_loss)
        self.train_cycle_count.assign_add(1.0)

        if True:
            self.all_opt_cycle_count.assign_add(1.0)
            dec_grads = grad_tape.gradient(
                elbo_rec_loss, self.decoder.trainable_variables
            )
            self.dec_opt.apply_gradients(
                zip(dec_grads, self.decoder.trainable_variables)
            )
            reg_grads = grad_tape.gradient(
                elbo_reg_loss, self.regressor.trainable_variables
            )
            self.reg_opt.apply_gradients(
                zip(reg_grads, self.regressor.trainable_variables)
            )
            enc_grads = grad_tape.gradient(total_loss, self.encoder.trainable_variables)
            self.enc_opt.apply_gradients(
                zip(enc_grads, self.encoder.trainable_variables)
            )
        else:
            self.single_opt_cycle_count.assign_add(1.0)
            enc_grads = grad_tape.gradient(total_loss, self.encoder.trainable_variables)
            self.enc_opt.apply_gradients(
                zip(enc_grads, self.encoder.trainable_variables)
            )

        all_activations = {"encoder": enc_act, "decoder": dec_act, "regressor": reg_act}
        del grad_tape
        self.update_trackers(
            elbo_rec_loss,
            elbo_reg_loss,
            rec_loss,
            kl_loss,
            kl_scale,
            reg_loss,
            child_genos,
            geno_logits,
            cur_epoch,
            pheno_pred,
            child_trait,
        )
        return all_activations

    @tf.function
    def test_step(self, data, cur_epoch):
        training = False
        geno_x, trait_x, meta_x = data
        seq_pos, chr_pos, pop_x = meta_x
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[:, :2, ...]
        child_genos = geno_x[:, 2, ...]

        mean, logvar, enc_act = self.encoder.call(
            geno_x, meta_x, training=training
        )
        # logvar = tf.constant(0.0)
        embed_x = self.sample_z(mean, logvar)
        geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
            parents_genos, embed_x, training=training
        )
        pheno_pred, reg_act, reg_gate = self.regressor.call(
            parent_trait, embed_x, parents_genos, training=training
        )
        kl_loss, reg_loss, rec_loss = self.loss_fn(
            child_genos,
            geno_logits,
            mean,
            logvar,
            trait_pred=pheno_pred,
            trait_true=child_trait,
            epoch=cur_epoch,
        )
        kl_scale = 1.0
        elbo_rec_loss = kl_loss * kl_scale + rec_loss
        elbo_reg_loss = kl_loss * kl_scale + reg_loss
        self.update_trackers(
            elbo_rec_loss,
            elbo_reg_loss,
            rec_loss,
            kl_loss,
            kl_scale,
            reg_loss,
            child_genos,
            geno_logits,
            cur_epoch,
            pheno_pred,
            child_trait,
        )

        return {}

    @property
    def metrics(self):
        return [
            self.kl_loss_tracker,
            self.kl_scale_tracker,
            self.elbo_rec_tracker,
            self.elbo_reg_tracker,
            self.reg_loss_tracker,
            self.epoch_tracker,
            self.cat_acc_tracker,
            self.reconstruction_loss_tracker,
            self.mean_loss_tracker,
            self.sd_loss_tracker,
        ] + self.class_acc_tracker
