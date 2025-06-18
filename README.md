[![arXiv](https://img.shields.io/badge/arXiv-2505.09306-b31b1b.svg)](https://arxiv.org/abs/2505.09306)
[![Dataset](https://img.shields.io/badge/dataset-available-4b44ce)](https://zenodo.org/records/15198884)
[![CI](https://github.com/vdplasthijs/PECL/actions/workflows/python-app.yml/badge.svg)](https://github.com/vdplasthijs/PECL/actions/workflows/python-app.yml)
![Issues](https://img.shields.io/github/issues/vdplasthijs/PECL)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
[![License](https://img.shields.io/github/license/vdplasthijs/PECL.svg)](LICENSE)

# Predicting butterfly species presence from satellite imagery using soft contrastive regularisation

This repository contains all code of our [2025 CVPR FGVC paper]([https://arxiv.org/abs/2505.09306](https://openaccess.thecvf.com/content/CVPR2025W/FGVC/html/Van_der_Plas_Predicting_butterfly_species_presence_from_satellite_imagery_using_soft_contrastive_CVPRW_2025_paper.html)), including:
- PECL (_Paired Embeddings Contrastive Loss_) implementation in `scripts/paired_embeddings_models.py`.
- Torch dataloader for the [S2BMS dataset](https://zenodo.org/records/15198884) in `scripts/DataSetImagePresence.py`
- Resnet-based model to predict species presence vectors from satellite images, using PECL.

### Installation:
- Use conda to install packages using `pecl.yml` or pip install from `requirements.txt`. 
- Add your user profile data paths in `content/data_paths_pecl.json`. (This step is not needed when just experimenting with the code and the example data provided in the repo). 

### Getting started:
- A sample data set (of 16 locations) is provided in `tests/data_tests/`.
- **Go to `notebooks/Getting started.ipynb` to see examples of how to load the data and model.**

### Data:
-  The full S2-BMS data set is available on [Zenodo](https://zenodo.org/records/15198884).
-  Our Torch dataloader is available in  `scripts/DataSetImagePresence.py`.

### PECL implementation
- For details please see our [paper](https://arxiv.org/abs/2505.09306).
- PyTorch implementation can be found in `scripts/paired_embeddings_models.py` (`ImageEncoder.pecl_loss()`).

### Results
-  The training scripts used for the paper are `scripts/train.py` and `scripts/train_randomsearch.py`.
-  The figures and tables in the paper were created in `notebooks/Results figs and tables.ipynb`. 


Please cite our [paper](https://arxiv.org/abs/2505.09306) if you use this method or data in a publication - thank you!!
