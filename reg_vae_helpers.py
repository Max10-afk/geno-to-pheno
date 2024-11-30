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



def plot_train_val_metrics(history, num_classes=11, suptitle="Model Performance Metrics"):
    """Plot all training and validation metrics in separate rows with a general title and class accuracy legend outside the plot."""
    
    # Define metric names
    general_metrics = ['reconstruction_loss', 'kl_loss', 'elbo', 'cat_acc']
    class_acc_metrics = [f'{i}_acc' for i in range(num_classes)]
    
    num_general_metrics = len(general_metrics)
    
    # Total number of columns
    total_cols = num_general_metrics + 1

    fig, axes = plt.subplots(2, total_cols, figsize=(total_cols * 5, 8), gridspec_kw={'hspace': 0.5})

    # General title
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.05)

    # Plot train metrics
    for i, metric in enumerate(general_metrics):
        ax = axes[1, i]
        ax.plot(history.history[metric], label=f'Train {metric}', color='blue')
        ax.set_title(f'Train {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True)

    # Plot train class accuracies
    ax = axes[1, -1]
    for class_acc in class_acc_metrics:
        if class_acc in history.history:
            ax.plot(history.history[class_acc], label=f'Train {class_acc}')
    ax.set_title('Train Class Accuracies')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot validation metrics
    for i, metric in enumerate(general_metrics):
        ax = axes[0, i]
        ax.plot(history.history.get(f'val_{metric}', []), label=f'Val {metric}', color='orange', linestyle='--')
        ax.set_title(f'Val {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True)

    # Plot val class accuracies
    ax = axes[0, -1]
    for class_acc in class_acc_metrics:
        val_acc = f'val_{class_acc}'
        if val_acc in history.history:
            ax.plot(history.history[val_acc], label=f'Val {class_acc}')
    ax.set_title('Val Class Accuracies')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to fit the suptitle and legend
    return fig, axes

freqs = np.unique(child_geno_np_train, return_counts = True)[1]/sum(np.unique(child_geno_np_train, return_counts = True)[1])

rec_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(from_logits = False, axis = -1,
            alpha = np.array([1, 0.01, 1, 1, 1, 0.01, 1, 1, 0.01, 1, 0.01]))
# rec_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = False, axis = -1)
test_var = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)
# ELBO Loss Layer
@keras.saving.register_keras_serializable()
class reg_vae_loss(layers.Layer):
    def call(self, x_labels, x_logits, trait_truth, trait_pred, epoch = None, step_size = 10):
        x_softmax = tf.nn.softmax(x_logits, axis = -1)
        rec_loss = rec_loss_fn(x_softmax, x_labels)
        kl_scale = 1
        kl_div = tf.keras.losses.KLD(x_labels, x_softmax) * kl_scale
        loss = tf.math.reduce_mean(rec_loss + kl_div)
        return loss, rec_loss, kl_div

@keras.saving.register_keras_serializable()
class feature_drop_layer(tf.keras.layers.Layer):
    def __init__(self, keep_prob = 0.25, feature_dim = 1, **kwargs):
        super().__init__()
        self.keep_prob = keep_prob
        self.feature_dim = feature_dim

    def call(self, inputs, training=True):
        if training:
            no_features = inputs.shape[self.feature_dim]
            feature_keep_bool = tf.ones(no_features) + tf.floor(tf.random.uniform([no_features]) - 0.25)
            reshape_dim = tf.concat([
                tf.ones(self.feature_dim, dtype=tf.int32), 
                [no_features], 
                tf.ones(tf.rank(inputs) - self.feature_dim - 1, dtype=tf.int32)
                ], axis=0)
            feature_keep_bool = tf.reshape(feature_keep_bool,
                                            reshape_dim)
            res = inputs * feature_keep_bool
            return res
        return inputs

def train_and_get_results(model,train_dataset = train_dataset, test_dataset = test_dataset, epochs = 100,
                            base_dir = "./data/var_autoencoder/",
                            files_to_backup = ["cur_helpers.py", "cur_encoder.py", "cur_decoder.py", "cur_autoencoder.py"],
                            write_to_disk = True):
    model_name = cur_base_dir.split("/")[-2]
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    for file in files_to_backup:
        shutil.copy("./" + file, base_dir)
    child_geno_data.iloc[~train_pops_idx, ].to_csv(base_dir + "child_genos_test.csv")
    child_geno_data.iloc[train_pops_idx, ].to_csv(base_dir + "child_genos_train.csv")

    p1_test_genos_df = pd.DataFrame(p1_genos_np_test)
    p1_test_genos_df["pop"] = p_pop_test
    p1_test_genos_df.to_csv(base_dir + "p1_test_genos_df.csv")

    p2_test_genos_df = pd.DataFrame(p2_genos_np_test)
    p2_test_genos_df["pop"] = p_pop_test
    p2_test_genos_df.to_csv(base_dir + "p2_test_genos_df.csv")

    p1_train_genos_df = pd.DataFrame(p1_genos_np_train)
    p1_train_genos_df["pop"] = p_pop_train
    p1_train_genos_df.to_csv(base_dir + "p1_train_genos_df.csv")

    p2_train_genos_df = pd.DataFrame(p2_genos_np_train)
    p2_train_genos_df["pop"] = p_pop_train
    p2_train_genos_df.to_csv(base_dir + "p2_train_genos_df.csv")

    model_train_loss = model.fit(train_dataset, epochs=epochs,
                                    validation_data = test_dataset)
    
    train_hist_df = pd.DataFrame(model_train_loss.history)
    train_hist_df["loss"] = model
    
    fig, axes = plot_train_val_metrics(model_train_loss, suptitle = "Model Performance Metrics using cross entropy loss")
    
    if write_to_disk:
        model.save(base_dir + "model.keras")
        train_hist_df.to_csv(base_dir + "train_hist.csv")
        fig.savefig(base_dir + "train_hist.png")
    return [model, train_hist_df, fig]