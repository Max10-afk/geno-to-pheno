
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

# rec_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(from_logits = False, axis = -1,
            #alpha = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]))
rec_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = False, axis = -1)
test_var = tf.Variable(initial_value=True, trainable=False, dtype=tf.bool)
# ELBO Loss Layer
@keras.saving.register_keras_serializable()
class elbo_loss(layers.Layer):
    def call(self, x_labels, x_logits, mean, logvar, no_kl, epoch = None, step_size = 10):
        x_softmax = tf.nn.softmax(x_logits, axis = -1)
        rec_loss = rec_loss_fn(x_softmax, x_labels)
        kl_scale = 1
        # kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1) * kl_scale
        kl_div = tf.keras.losses.KLD(x_labels, x_softmax) * kl_scale
        no_kl_tensor = tf.convert_to_tensor(no_kl, dtype=tf.bool)
        # loss = tf.cond(
        #     pred = no_kl,
        #     true_fn = lambda: tf.math.reduce_mean(rec_loss),
        #     false_fn = lambda: tf.math.reduce_mean(rec_loss + kl_div)
        # )
        loss = tf.math.reduce_mean(rec_loss + kl_div)
        # loss = tf.case([(tf.convert_to_tensor(test_var), lambda: tf.math.reduce_mean(rec_loss))], default= lambda: tf.math.reduce_mean(rec_loss + kl_div))
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

def train_and_get_results(model, desc_str, train_dataset = train_dataset, test_dataset = test_dataset, epochs = 100,
                            base_dir = "./data/var_autoencoder/",
                            files_to_backup = ["helpers.py", "encoder.py", "decoder.py", "autoencoder.py"]):
    out_folder = base_dir + desc_str + "/"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    for file in files_to_backup:
        shutil.copy("./" + file, out_folder)
    child_geno_data.iloc[~train_pops_idx, ].to_csv(out_folder + "child_genos_test.csv")
    child_geno_data.iloc[train_pops_idx, ].to_csv(out_folder + "child_genos_train.csv")

    p1_test_genos_df = pd.DataFrame(p1_genos_np_test)
    p1_test_genos_df["pop"] = p_pop_test
    p1_test_genos_df.to_csv(out_folder + "p1_test_genos_df.csv")

    p2_test_genos_df = pd.DataFrame(p2_genos_np_test)
    p2_test_genos_df["pop"] = p_pop_test
    p2_test_genos_df.to_csv(out_folder + "p2_test_genos_df.csv")

    p1_train_genos_df = pd.DataFrame(p1_genos_np_train)
    p1_train_genos_df["pop"] = p_pop_train
    p1_train_genos_df.to_csv(out_folder + "p1_train_genos_df.csv")

    p2_train_genos_df = pd.DataFrame(p2_genos_np_train)
    p2_train_genos_df["pop"] = p_pop_train
    p2_train_genos_df.to_csv(out_folder + "p2_train_genos_df.csv")

    model_train_loss = model.fit(train_dataset, epochs=epochs,
                                    validation_data = test_dataset)
    model.save(out_folder + desc_str + ".keras")
    train_hist_df = pd.DataFrame(model_train_loss.history)
    train_hist_df["loss"] = desc_str
    train_hist_df.to_csv(out_folder + desc_str + ".csv")
    fig, axes = plot_train_val_metrics(model_train_loss, suptitle = "Model Performance Metrics using cross entropy loss")
    fig.savefig(out_folder + desc_str + ".png")
