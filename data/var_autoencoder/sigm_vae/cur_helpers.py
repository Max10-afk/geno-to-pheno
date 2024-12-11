
def plot_train_val_metrics(history, num_classes=11, suptitle="Model Performance Metrics"):
    """Plot all training and validation metrics in separate rows with a general title and class accuracy legend outside the plot."""
    
    # Define metric names
    general_metrics = ['reconstruction_loss', 'kl_loss', 'elbo', 'cat_acc', 'kl_scale']
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
        ax.plot(history[metric], label=f'Train {metric}', color='blue')
        ax.set_title(f'Train {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True)

    # Plot train class accuracies
    ax = axes[1, -1]
    for class_acc in class_acc_metrics:
        if class_acc in history:
            ax.plot(history[class_acc], label=f'Train {class_acc}')
    ax.set_title('Train Class Accuracies')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot validation metrics
    for i, metric in enumerate(general_metrics):
        ax = axes[0, i]
        ax.plot(history.get(f'val_{metric}', []), label=f'Val {metric}', color='orange', linestyle='--')
        ax.set_title(f'Val {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True)

    # Plot val class accuracies
    ax = axes[0, -1]
    for class_acc in class_acc_metrics:
        val_acc = f'val_{class_acc}'
        if val_acc in history:
            ax.plot(history[val_acc], label=f'Val {class_acc}')
    ax.set_title('Val Class Accuracies')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust layout to fit the suptitle and legend
    return fig, axes

freqs = np.unique(child_geno_np_train, return_counts = True)[1]/sum(np.unique(child_geno_np_train, return_counts = True)[1])

inf_w = 1
freq_w = 8e-1
rec_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(from_logits = False, axis = -1,
           alpha = np.array([inf_w, freq_w, inf_w, inf_w, inf_w, freq_w, inf_w, inf_w, freq_w, inf_w, freq_w]),
           label_smoothing = 0.1)
# rec_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = False, axis = -1,
#     label_smoothing = 0.4)
# rec_loss_fn = tf.keras.losses.KLDivergence()

# ELBO Loss Layer
@keras.saving.register_keras_serializable()
class elbo_loss(layers.Layer):
    @tf.function
    def call(self, x_labels, x_logits, mean, logvar, no_kl, epoch = None, step_size = 50):
        x_softmax = tf.nn.softmax(x_logits, axis = -1)
        kl_scale = 1
        if not epoch is None:
            #kl_scale = tf.math.maximum(0.5, (tf.sin(epoch * 0.05 - 1/2 * tf.constant(m.pi)) + 1) / 2)
            # kl_scale = (tf.sin(epoch * 0.05 - 1/2 * tf.constant(m.pi)) + 1) / 2
            kl_scale = 1/(1 + tf.math.exp(13 - 0.1 * epoch))
        rec_loss = rec_loss_fn(x_labels, x_softmax)
        kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
        loss = tf.math.reduce_mean(rec_loss + kl_div * kl_scale)
        return loss, rec_loss, kl_div, kl_scale

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

# Training loop function
def train_autoencoder(model, train_dataset, val_dataset, epochs, base_dir,
        backup_epoch = 50):
    full_log = {cur_metric.name: np.empty(shape=(1)) for cur_metric in model.metrics}
    train_log = {cur_metric.name: np.empty(shape=(1)) for cur_metric in model.metrics}
    val_log = {"val_" + cur_metric.name: np.empty(shape=(1)) for cur_metric in model.metrics}
    cur_epoch_tf = tf.Variable(initial_value = 0., trainable = False)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        cur_epoch_tf.assign(float(epoch))
        # Reset the states of the metrics
        model.reset_metrics()
        # Training loop
        for step, train_step_data in enumerate(train_dataset):
            model.train_step(train_step_data, cur_epoch = cur_epoch_tf)
        
        # Collect mean metrics at the end of the epoch for training
        train_metrics = {model.metrics[cur_id].name:model.metrics[cur_id].result() for cur_id in range(len(model.metrics))}
        train_log = {cur_metric: np.append(train_log[cur_metric], train_metrics[cur_metric].numpy()) for cur_metric in train_log.keys()}
        # Reset the states of the metrics for validation
        model.reset_metrics()
        
        # Validation loop
        for val_step, val_step_data in enumerate(val_dataset):
            model.test_step(val_step_data, cur_epoch = cur_epoch_tf)
        
        # Collect mean metrics at the end of the epoch for validation
        val_metrics = {model.metrics[cur_id].name:model.metrics[cur_id].result() for cur_id in range(len(model.metrics))}
        val_log = {cur_metric: np.append(val_log[cur_metric], val_metrics[cur_metric.replace("val_", "")].numpy())
            for cur_metric in val_log.keys()}
        # Print collected mean metrics
        print(f"Epoch {epoch+1} train metrics:")
        for cur_metric_name in train_metrics.keys():
            print(f"{cur_metric_name}: {train_metrics[cur_metric_name]}", end = ", ")
        print("")
        print(f"Epoch {epoch+1} val metrics:")
        for cur_metric_name in val_metrics.keys():
            print(f"{cur_metric_name}: {val_metrics[cur_metric_name]}", end = ", ")
        if (epoch % backup_epoch) == 0:
            full_log.update(train_log)
            full_log.update(val_log)
            train_hist_df = pd.DataFrame(full_log)
            # train_hist_df["loss"] = model
            train_hist_df.to_csv(f"{base_dir}train_hist_epoch_{epoch}.csv")
            tf.keras.backend.clear_session()
            gc.collect()
    full_log.update(train_log)
    full_log.update(val_log)
    return full_log

def train_and_get_results(model, train_dataset = train_dataset, test_dataset = test_dataset, epochs = 100,
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
    # def train_autoencoder(model, train_dataset, val_dataset, epochs):
    model_train_loss = train_autoencoder(model, train_dataset, val_dataset = test_dataset,
        epochs=epochs, base_dir = base_dir)
    train_hist_df = pd.DataFrame(model_train_loss)
    train_hist_df["loss"] = model
    
    fig, axes = plot_train_val_metrics(model_train_loss, suptitle = "Model Performance Metrics using cross entropy loss")
    
    if write_to_disk:
        model.save(base_dir + "model.keras")
        train_hist_df.to_csv(base_dir + "train_hist.csv")
        fig.savefig(base_dir + "train_hist.png")
    return [model, train_hist_df, fig]