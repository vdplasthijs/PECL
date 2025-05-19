# Predicting butterfly species presence from satellite data

Resnet-based model to predict species presence vectors from satellite images. The model uses PECL (_Paired Embeddings Contrastive Loss_) as contrastive regularisation, which is detailed in our 2025 CVPR FGVC paper:

[paper](https://arxiv.org/abs/2505.09306), [data](https://zenodo.org/records/15198884), [code](https://github.com/vdplasthijs/PECL). 

### Installation:
- Use conda to install packages using `pecl.yml` or pip install from `requirements.txt`. 
- Add your user profile data paths in `content/data_paths_pecl.json`. (This step is not needed when just experimenting with the code and the example data provided in the repo). 

### Getting started:
- An example data set is provided in `tests/data_tests/`.
- **Go to `notebooks/Getting started.ipynb` to see examples of how to load the data and model.**
-  The full S2-BMS data set is available on [Zenodo](https://zenodo.org/records/15198884).

### PECL implementation
- For details please see our [paper](https://arxiv.org/abs/2505.09306).
- PyTorch implementation can be found in `scripts/paired_embeddings_models.py` (`ImageEncoder.pecl_loss()`).
- Models are trained by running `scripts/train.py` and `scripts/train_randomsearch.py`.

### Results
- The `notebooks/` folder contains the notebooks for creating figures/tables. 


Please cite our [paper](https://arxiv.org/abs/2505.09306) if you use this method or data in a publication - thank you!!