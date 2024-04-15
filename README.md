# Predicting butterfly species presence from satellite data

Resnet-based model to predict species presence vectors from satellite images. The model uses PECL (_Paired Embeddings Contrastive Loss_) as contrastive regularisation. More details to be added following an upcoming publication. 

### Installation:
- Use conda to install packages using `pecl.yml`. 
- Add your user profile data paths in `content/data_paths_pecl.json`. 

### Loading the data:
- A link to the S2-BMS data set will be added soon.
- A custom DataSet class is provided in `scripts/DataSetImagePresence.py`

### PECL implementation
- Details will follow in the upcoming publication.
- PyTorch implementation can be found in `scripts/paired_embeddings_models.py` (`ImageEncoder.pecl_loss()`).
- Models are trained by running `scripts/train.py` and `scripts/train_randomsearch.py`.

### Results
- The `notebooks/` folder contains the notebooks for creating figures/tables. 
