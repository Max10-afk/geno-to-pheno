def perc_error(y_true, y_pred):
    diff = tf.math.abs((y_true - y_pred)/y_true)
    n_samples = tf.cast(tf.shape(diff)[0], dtype=tf.float64)
    mean_per_f = tf.math.reduce_sum(diff, axis = 0)/n_samples
    return mean_per_f

def error_per_feature_mean(y_true, y_pred):
    return perc_error((tf.cast(y_true[:, 0], tf.float64)), tf.cast(y_pred[:, 0], tf.float64))

def error_per_feature_sd(y_true, y_pred):
    return perc_error((tf.cast(y_true[:, 1], tf.float64)), tf.cast(y_pred[:, 1], tf.float64))

def mean_y(y_true, y_pred):
    return tf.math.reduce_mean(y_pred)

def std_y(y_true, y_pred):
    return tf.math.reduce_std(y_pred)

@tf.function
def sample_z(mean, logvar):
    batch_size = tf.shape(mean)[0]  # Use dynamic shape inference for batch size
    latent_dim = mean.shape[1]      # Latent dimension can be known statically
    eps = tf.random.normal(shape=(batch_size, latent_dim))  # Use dynamic batch size
    res = eps * tf.exp(logvar * .5) + mean
    return res

@keras.saving.register_keras_serializable()
class sampling_layer(tf.keras.layers.Layer):
    def call(self, inputs, training=True):
        if training:
            return sample_z(inputs[:, 0, ...], inputs[:, 1, ...])
        return inputs