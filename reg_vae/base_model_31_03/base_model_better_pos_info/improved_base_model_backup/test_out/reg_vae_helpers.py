## Model functions/variables
fc_reg = keras.regularizers.L2(1e-2)
inf_w = 1
freq_w = 8e-1
ACT_LAYER = tf.keras.layers.LeakyReLU(alpha=0.5)
# rec_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(from_logits = False, axis = -1,
#            alpha = np.array([inf_w, freq_w, inf_w, inf_w, inf_w, freq_w, inf_w, inf_w, freq_w, inf_w, freq_w]),
#            label_smoothing = 0.1)
rec_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
reg_loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
# rec_loss_fn = tf.keras.losses.KLDivergence()
@tf.function
def cantor_pairing(inp_vec):
    input = tf.sort(inp_vec, axis = 1)
    a = input[:, 0]
    b = input[:, 1]
    return 0.5 * (a + b) * (a + b + 1) + b
def inverse_cantor_pairing_tf(z):
    w = tf.floor((tf.sqrt(8 * z + 1) - 1) / 2)
    t = (w * (w + 1)) / 2
    y = z - t
    x = w - y
    return tf.cast(x, dtype=tf.int32), tf.cast(y, dtype=tf.int32)


def positional_encoding(position, d_model):
 
    # Create a matrix of shape [position, d_model] where each element is the position index
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    
    # Apply sine to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply cosine to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)


def embed_pop_distances(embed, pop_map):
    # calculate pairwise distances between embeddings of individuals of the same population
    # and return the mean distance for each population using tensor operations
    # Get the unique populations
    unique_pops = tf.unique(pop_map)[0]
    all_distances = tf.expand_dims(embed, axis=1) - tf.expand_dims(embed, axis=0)
    mask = tf.eye(tf.shape(all_distances)[0], dtype=tf.bool)
    all_distances = tf.boolean_mask(all_distances, ~mask)
    # Calculate the pairwise all_distances
    all_distances = tf.norm(all_distances, axis=-1)
    mean_all_distances = tf.reduce_mean(all_distances)
    # Initialize a list to store the mean distances for each population
    mean_distances = []
    # Loop through each unique population
    for pop in unique_pops:
        # Get the indices of individuals in the current population
        pop_indices = tf.where(pop_map == pop)[:, 0]
        # Get the embeddings for individuals in the current population
        if len(pop_indices) < 2:
            continue
        pop_embeddings = tf.gather(embed, pop_indices)
        # Calculate pairwise distances using broadcasting
        distances = tf.expand_dims(pop_embeddings, axis=1) - tf.expand_dims(pop_embeddings, axis=0)
        # remove diagonal elements (self-distances)
        mask = tf.eye(tf.shape(pop_embeddings)[0], dtype=tf.bool)
        distances = tf.boolean_mask(distances, ~mask)
        # Calculate the pairwise distances
        distances = tf.norm(distances, axis=-1)
        # Calculate the mean distance for the current population
        mean_distance = tf.reduce_mean(distances)
        mean_distances.append(mean_distance)
    # Convert the list of mean distances to a tensor
    mean_distances = tf.stack(mean_distances)
    # Calculate the mean distance across all populations
    mean_distance = tf.reduce_mean(mean_distances)
    return mean_distance / mean_all_distances

def inter_pop_dist(input, pop_map, loss_fn):
    # calculate pairwise distances between embeddings of individuals of the same population
    # and return the mean distance for each population using tensor operations
    # Get the unique populations
    unique_pops = tf.unique(pop_map)[0]
    all_distances = loss_fn(tf.expand_dims(embed, axis=1), tf.expand_dims(embed, axis=0))
    mask = tf.eye(tf.shape(all_distances)[0], dtype=tf.bool)
    all_distances = tf.boolean_mask(all_distances, ~mask)
    # Calculate the pairwise all_distances
    all_distances = tf.norm(all_distances, axis=-1)
    mean_all_distances = tf.reduce_mean(all_distances)
    # Initialize a list to store the mean distances for each population
    mean_distances = []
    # Loop through each unique population
    for pop in unique_pops:
        # Get the indices of individuals in the current population
        pop_indices = tf.where(pop_map == pop)[:, 0]
        # Get the embeddings for individuals in the current population
        if len(pop_indices) < 2:
            continue
        pop_embeddings = tf.gather(embed, pop_indices)
        # Calculate pairwise distances using broadcasting
        distances = loss_fn(tf.expand_dims(pop_embeddings, axis=1), tf.expand_dims(pop_embeddings, axis=0))
        # remove diagonal elements (self-distances)
        mask = tf.eye(tf.shape(pop_embeddings)[0], dtype=tf.bool)
        distances = tf.boolean_mask(distances, ~mask)
        # Calculate the pairwise distances
        distances = tf.norm(distances, axis=-1)
        # Calculate the mean distance for the current population
        mean_distance = tf.reduce_mean(distances)
        mean_distances.append(mean_distance)
    # Convert the list of mean distances to a tensor
    mean_distances = tf.stack(mean_distances)
    # Calculate the mean distance across all populations
    mean_distance = tf.reduce_mean(mean_distances)
    return mean_distance / mean_all_distances


def new_positional_encoding(positions, d_model):
    # Convert positions to a float32 tensor if not already
    positions = tf.convert_to_tensor(
        tf.cast(positions, dtype=tf.float32),
        dtype=tf.float32)
    
    # Expand positions to shape [num_positions, 1]
    positions_expanded = tf.expand_dims(positions, axis=1)
    
    # Create dimension indices and calculate divisor terms
    i = tf.range(d_model, dtype=tf.float32)
    div_term = tf.cast((i // 2) * 2, tf.float32)/ tf.cast(d_model, tf.float32)
    
    # Compute denominator with broadcasting
    denominator = tf.pow(10000.0, div_term)
    denominator = tf.expand_dims(denominator, axis=0)  # Shape [1, d_model]
    
    # Calculate angle values
    angle_rads = positions_expanded / denominator
    
    # Create mask for even indices and interleave sin/cos
    even_mask = tf.range(d_model) % 2 == 0
    pos_encoding = tf.where(even_mask, tf.sin(angle_rads), tf.cos(angle_rads))
    
    # Add batch dimension for consistency with original format
    return tf.expand_dims(pos_encoding, axis=0)

# ELBO Loss Layer
@tf.keras.utils.register_keras_serializable()
class elbo_loss(layers.Layer):
    # @tf.function
    def call(
        self,
        x_labels,
        x_logits,
        mean,
        logvar,
        trait_pred,
        trait_true,
        epoch=None,
        step_size=50,
    ):
        x_softmax = tf.nn.softmax(x_logits, axis=-1)
        reg_loss = reg_loss_fn(trait_true, trait_pred)
        print("reg_loss: ", reg_loss)
        rec_loss = rec_loss_fn(x_labels, x_softmax)
        print("rec_loss: ", rec_loss)
        kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
        print("kl_div: ", kl_div)
        # kl_div pushing towards mean of 1, avoiding negative values in latent space:
        # kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean - 1) - tf.exp(logvar), axis=-1)
        return kl_div, reg_loss, rec_loss


# Variance Importance Callback
class VarImpVIANN(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0
        self.w_shape = None

    def on_train_begin(self, logs={}, verbose=1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.w_shape = logs["w_shape"]
        #   print("VarImpVIANN layer name ", self.model.layers[0].layers[0].name)
        w = tf.reduce_sum(self.model.layers[0].get_weights()[0], axis=-1)
        # print(self.model.layers[0].get_weights()[0].shape)
        # print("Reduced sum weights shape: ", w.shape)
        # print("target reshaped weights shape: ", self.w_shape)
        w = tf.reshape(w, self.w_shape)
        # print("Reshaped weights shape: ", w.shape)
        self.diff = tf.reduce_sum(w, axis=-1)
        # print("Initial diff shape: ", self.diff.shape)

    def on_epoch_end(self, batch, logs={}):
        # Sum over dense layer feature axis -> obtain summed weights per flattened input feature
        w = tf.reduce_sum(self.model.layers[0].get_weights()[0], axis=-1)
        # print("on_epoch_end Reduced sum weights shape: ", w.shape)
        # print("on_epoch_end target reshaped weights shape: ", self.w_shape)
        # reshape w to reconstruct weight per snp feature
        w = tf.reshape(w, self.w_shape)
        # sum over snp one hot encoding dimension and parental dimension
        w = tf.reduce_sum(w, axis=-1)
        self.n += 1
        delta = np.subtract(w, self.diff)
        self.diff += delta / self.n
        delta2 = np.subtract(w, self.diff)
        self.M2 += delta * delta2

        self.lastweights = w

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float("nan")
        else:
            self.s2 = self.M2 / (self.n - 1)

        scores = np.multiply(self.s2, np.abs(self.lastweights))
        print(f"Final variable importance scores shape: {scores.shape}")

        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        if self.verbose:
            print(
                "Most important variables: ",
                np.array(self.varScores).argsort()[-10:][::-1],
            )


@tf.keras.utils.register_keras_serializable()
class feature_drop_layer(tf.keras.layers.Layer):
    def __init__(self, keep_prob=0.25, feature_dim=1, **kwargs):
        super().__init__()
        self.keep_prob = keep_prob
        self.feature_dim = feature_dim

    def call(self, inputs, training):
        if training:
            no_features = inputs.shape[self.feature_dim]
            feature_keep_bool = tf.ones(no_features) + tf.floor(
                tf.random.uniform([no_features]) - 0.25
            )
            reshape_dim = tf.concat(
                [
                    tf.ones(self.feature_dim, dtype=tf.int32),
                    [no_features],
                    tf.ones(tf.rank(inputs) - self.feature_dim - 1, dtype=tf.int32),
                ],
                axis=0,
            )
            feature_keep_bool = tf.reshape(feature_keep_bool, reshape_dim)
            res = inputs * feature_keep_bool
            return res
        return inputs


## General Helper functions
def train_loop(
    model,
    train_dataset,
    val_dataset,
    epochs,
    track_activations=False,
    epoch_save_interval=10,
    step_save_interval=2,
    _callbacks=[],
):
    for batch in train_dataset:
        test_batch = batch
        break
    geno_shape = test_batch[0][1].shape[1:]
    print(f"geno_shape: {geno_shape}")
    callbacks = tf.keras.callbacks.CallbackList(
        _callbacks, add_history=True, model=model
    )
    logs = {"w_shape": geno_shape}
    callbacks.on_train_begin(logs=logs)
    train_log = {cur_metric.name: np.empty(shape=(1)) for cur_metric in model.metrics}
    val_log = {
        "val_" + cur_metric.name: np.empty(shape=(1)) for cur_metric in model.metrics
    }
    cur_epoch_tf = tf.Variable(initial_value=0.0, trainable=False)
    act_tracker = {}
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch, logs=logs)
        cur_epoch_tf.assign(float(epoch))
        # Reset the states of the metrics
        model.reset_metrics()
        # Training loop
        sub_act_tracker = {}
        for step, train_step_data in enumerate(train_dataset):
            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_train_batch_begin(step, logs=logs)
            if (
                (epoch - 1) % epoch_save_interval == 0
                and step % step_save_interval == 0
                and track_activations
            ):
                print(f"Step {step}, tracking activations")
                _, activations = model.train_step(
                    train_step_data, cur_epoch=cur_epoch_tf, return_activations=True
                )
                for key in activations.keys():
                    activations[key] = {
                        cur_key: activations[key][cur_key].numpy()
                        for cur_key in activations[key].keys()
                    }
                sub_act_tracker[step] = activations
            else:
                model.train_step(
                    train_step_data, cur_epoch=cur_epoch_tf, return_activations=False
                )
            callbacks.on_train_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)
        if (epoch - 1) % 10 == 0 and track_activations:
            act_tracker[epoch] = sub_act_tracker
        # Collect mean metrics at the end of the epoch for training
        train_metrics = {
            model.metrics[cur_id].name: model.metrics[cur_id].result()
            for cur_id in range(len(model.metrics))
        }
        train_log = {
            cur_metric: np.append(
                train_log[cur_metric], train_metrics[cur_metric].numpy()
            )
            for cur_metric in train_log.keys()
        }
        # Reset the states of the metrics for validation
        model.reset_metrics()

        # Validation loop
        for val_step, val_step_data in enumerate(val_dataset):
            callbacks.on_batch_begin(val_step, logs=logs)
            callbacks.on_test_batch_begin(val_step, logs=logs)
            _ = model.test_step(val_step_data, cur_epoch=cur_epoch_tf)
            callbacks.on_test_batch_end(val_step, logs=logs)
            callbacks.on_batch_end(val_step, logs=logs)
        # Collect mean metrics at the end of the epoch for validation
        val_metrics = {
            model.metrics[cur_id].name: model.metrics[cur_id].result()
            for cur_id in range(len(model.metrics))
        }
        val_log = {
            cur_metric: np.append(
                val_log[cur_metric], val_metrics[cur_metric.replace("val_", "")].numpy()
            )
            for cur_metric in val_log.keys()
        }
        # Print collected mean metrics
        print(f"Epoch {epoch+1} train metrics:")
        for cur_metric_name in train_metrics.keys():
            print(f"{cur_metric_name}: {train_metrics[cur_metric_name]}", end=", ")
        print("")
        print(f"Epoch {epoch+1} val metrics:")
        for cur_metric_name in val_metrics.keys():
            print(f"{cur_metric_name}: {val_metrics[cur_metric_name]}", end=", ")
        if (epoch % 100) == 0:
            tf.keras.backend.clear_session()
            gc.collect()
        callbacks.on_epoch_end(epoch, logs=logs)
    callbacks.on_train_end(logs=logs)
    train_log.update(val_log)
    # history_object = None
    # for cb in callbacks:
    #     if isinstance(cb, tf.keras.callbacks.History):
    #         history_object = cb
    return train_log, act_tracker, callbacks


def train_and_get_results(
    model,
    train_dataset,
    test_dataset,
    epochs=100,
    base_dir="./data/var_autoencoder/",
    files_to_backup=[
        "cur_helpers.py",
        "cur_encoder.py",
        "cur_decoder.py",
        "cur_autoencoder.py",
    ],
    write_to_disk=True,
    track_activations=False,
    epoch_save_interval=10,
    step_save_interval=2,
    callbacks=[],
    save_train_data=False,
    train_idx=None,
    eval_idx=None,
    test_idx=None,
    file_dir="./",
):

    model_name = cur_base_dir.split("/")[-2]
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    for file in files_to_backup:
        shutil.copy(file_dir + file, base_dir)
    child_data_labels = np.array(np.arange(child_geno_data.shape[0]), dtype="str")
    child_data_labels[train_idx] = "train"
    child_data_labels[eval_idx] = "eval"
    child_data_labels[test_idx] = "test"
    child_geno_data_c = child_geno_data
    child_geno_data_c["d_type"] = child_data_labels
    child_geno_data_c.to_csv(base_dir + "child_geno_data.csv")
    parental_geno_data.to_csv(base_dir + "parental_geno_data.csv")
    child_pheno_data_c = child_pheno_data
    child_pheno_data_c["d_type"] = child_data_labels
    child_pheno_data_c.to_csv(base_dir + "child_pheno_data.csv")
    parent_pheno_data.to_csv(base_dir + "parent_pheno_data.csv")
    child_geno_data_c = child_geno_data
    child_geno_data_c["d_type"] = child_data_labels
    child_geno_data_c.to_csv(base_dir + "child_geno_data.csv")

    model_train_loss, all_activations, callbacks_out = train_loop(
        model,
        train_dataset,
        val_dataset=test_dataset,
        epochs=epochs,
        track_activations=track_activations,
        epoch_save_interval=epoch_save_interval,
        step_save_interval=step_save_interval,
        _callbacks=callbacks,
    )
    train_hist_df = pd.DataFrame(model_train_loss)
    train_hist_df["loss"] = model

    fig, axes = plot_train_val_metrics(
        model_train_loss, suptitle="Model Performance Metrics using cross entropy loss"
    )

    if write_to_disk:
        model.save(base_dir + "model.keras")
        train_hist_df.to_csv(base_dir + "train_hist.csv")
        fig.savefig(base_dir + "train_hist.png")
        # if track_activations:
        #     with open(base_dir + "activations.pkl", "wb") as f:
        #         pickle.dump(all_activations, f)
    return [model, train_hist_df, fig, all_activations, callbacks_out]


def plot_train_val_metrics(
    history, num_classes=11, suptitle="Model Performance Metrics"
):
    """Plot all training and validation metrics in separate rows with a general title and class accuracy legend outside the plot."""

    # Define metric names
    general_metrics = [
        "reconstruction_loss",
        "reg_loss",
        "kl_loss",
        "elbo_rec",
        "elbo_reg",
        "cat_acc",
        "kl_scale",
        "mean_deviation",
        "sd_deviation",
    ]
    class_acc_metrics = [f"{i}_acc" for i in range(num_classes)]

    num_general_metrics = len(general_metrics)

    # Total number of columns
    total_cols = num_general_metrics + 1

    fig, axes = plt.subplots(
        2, total_cols, figsize=(total_cols * 5, 8), gridspec_kw={"hspace": 0.5}
    )

    # General title
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.05)

    # Plot train metrics
    for i, metric in enumerate(general_metrics):
        ax = axes[1, i]
        ax.plot(history[metric], label=f"Train {metric}", color="blue")
        ax.set_title(f"Train {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)

    # Plot train class accuracies
    ax = axes[1, -1]
    for class_acc in class_acc_metrics:
        if class_acc in history:
            ax.plot(history[class_acc], label=f"Train {class_acc}")
    ax.set_title("Train Class Accuracies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Plot validation metrics
    for i, metric in enumerate(general_metrics):
        ax = axes[0, i]
        ax.plot(
            history.get(f"val_{metric}", []),
            label=f"Val {metric}",
            color="orange",
            linestyle="--",
        )
        ax.set_title(f"Val {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)

    # Plot val class accuracies
    ax = axes[0, -1]
    for class_acc in class_acc_metrics:
        val_acc = f"val_{class_acc}"
        if val_acc in history:
            ax.plot(history[val_acc], label=f"Val {class_acc}")
    ax.set_title("Val Class Accuracies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(
        rect=[0, 0, 0.85, 0.95]
    )  # Adjust layout to fit the suptitle and legend
    return fig, axes
