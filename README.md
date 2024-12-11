# Geno to Pheno Repo

Quick explainer for the fodler structure:
* cur_\* files are model tensorflow definitions, some of them outdated/experimental
  * VAE specific files are:
    * cur_encoder: encoder model
    * cur_decoder: decoder model
    * cur_autoencoder: model stiching together encoder + decoder, additionally defines train/test_step
    * custom_lr_on_callback: custom implementation of ReduceLROnPlateau with additional start_epoch parameter
    * cur_helpers: helper functions such as custom model.fit function, metric visualisations and loss function
    * var_autoencoder_parental.ipynb: notebook where actual VAE training & data extraction takes place
    * data/var_autoencoder/: Different VAE implementation backups (definitions, weights & data used for training)
  * Geno to trait specific files are:
    * geno_to_trait_model.py: tf model definitions
    * geno_to_trait_helpers.py: helper functions such as sampleing layers and functions used to record metrics
    * geno_to_trait.ipynb: notebook where actual training & data extraction takes place
  * reg VAE specific files are:
    * reg_vae_helpers.py: helper functions such as custom model.fit function, metric visualisations and loss function
    * cur_encoder.py: encoder model
    * cur_decoder.py: decoder model
    * reg_autoencoder.py: trash, to be deleted
    * geno_to_trait_model.py: trait prediction model
    * regression_vae.py: model stiching together encoder + decoder + geno_to_trait, additionally defines train/test_step
    * data/reg_vae/: Different reg VAE implementation backups (definitions, weights & data used for training)

Other folders than those mentioned previously in /data/ mostly contain raw data files, actual data used for training are in the .csv files in /data.
Note that .keras weight files are to large for github, therefore i cant upload them :(