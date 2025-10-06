# LNPDB: Lipid Nanoparticle Database towards structure-function modeling and data-driven design for nucleic acid delivery

<p align="center">
  <img src="data/misc/LNPDB_fig1.png" alt="alt text" width="800px" align="middle"/>
</p>

*Figure: LNPDB (https://lnpdb.molcube.com) is a data repository and web tool for compiling and uploading LNP structure-function data.*

## Description

This repository primarily includes the database itself and code to run deep learning models and analyze molecular dynamics trajectories, as detailed in the paper "LNPDB: Lipid Nanoparticle Database towards structure-function modeling and data-driven design for nucleic acid delivery" (*In revision*).

The entire LNPDB can be found here in this repo in `/data/LNPDB_for_LiON/LNPDB.csv`; however, we encourage users to instead visit https://lnpdb.molcube.com to interactively search and download the database.

_____

# Table of contents

- [Repository structure](#repository-structure)
- [System requirements](#system-requirements)
- [Installation guide](#installation-guide)
- [Instructions](#instructions)
- [Citation](#citation)

_____

# Repository structure

The repository is structured as follows:

- `/data`: contains example molecular dynamics trajectory data and implementation of LiON for LNPDB.
  - `/MD_trajectory_neutral`: this folder is currently blank due to file size limitations. Download one representative trajectory [here](https://www.dropbox.com/scl/fi/wtq4asngi2t8v7ij1suqw/MD_trajectory_neutral.zip?rlkey=xj76yl9r3zy1i9cel7rubz1a2&st=qcayrgmo&dl=0).
  - `/MD_trajectory_protonated`: this folder is currently blank due to file size limitations. Download one representative trajectory [here](https://www.dropbox.com/scl/fi/4ec29ze4riamqjbj3c51f/MD_trajectory_protonated.zip?rlkey=0ed57ot7qyae39gxvi3ufq5ey&st=eun40vkr&dl=0).
  - `/LNPDB_for_LiON`: this folder contains LNPDB organized to be trained & tested by the LiON deep learning model to predict LNP delivery efficacy. LNPDB is provided as `/single_split` (70%-15%-15% train-validation-test) as evaluated in Fig. 3a, and 5 `/cv_splits` (80%-20% train-test) as evaluated in Fig. 3b. This folder already contains the LiON trained model checkpoints.
  - `/LNPDB_for_AGILE`: this folder primarily contains LNPDB heldout data to be tested by the AGILE deep learning model to predict LNP delivery efficacy.
  
- `/results`: contains results for the different measures derived from molecular dynamics trajectories.

- `/scripts`: contains Python scripts used to process data and generate results.

Note that the first and last time point MD trajectory files for all simulated systems included in our paper (Table S2) can be downloaded [here](https://www.dropbox.com/scl/fi/c75w2nty2567ymftyim9x/MD_trajectory_ini_last.zip?rlkey=hn1gr8wokquyahl1bkcbibg4u&st=6jy8af1l&dl=0).

_____

# System requirements

## Hardware requirements

To run the analysis code in this repository, it is recommended to have a computer with enough RAM (> 8 GB) to support the in-memory operations. Moreover, robust GPUs are recommended to run the all-atom simulations.

## Software requirements

This code has been implemented with Python version 3.9.20. Other Python versions will likely work. See below for creating an appropriate conda environment.

### OS requirements

This code has been tested on the following systems, although it should work generally for other OS types (e.g., Windows) with potentially minor required changes to package versions.

- macOS: Ventura 13.2.1

- Linux: Ubuntu 22.04

____

# Installation guide

## Install repository

```
git clone https://github.com/evancollins1/LNPDB.git
```

## Install Python modules

Create a conda environment (`lnpdb`) with the modules specified in `environment.yml`.

```
conda env create -f environment.yml
conda activate lnpdb
```

____


# Instructions

## Analyzing MD trajectories

To analyze the provided molecular dynamics trajectories, follow along the Jupyter notebooks in the `/scripts` folder, i.e., `analyze_CPP_Rg.ipynb`, `analyze_CPP_V.ipynb`, and `analyze_other_metrics.ipynb`.

## Training & testing LiON deep learning model

To train the LiON deep learning model ([Witten et al., *Nat Biotech*, 2024](https://www.nature.com/articles/s41587-024-02490-y)) on the LNP formulations of LNPDB, LNPDB is provided as `/single_split` (70%-15%-15% train-validation-test) as evaluated in Fig. 3a, and 5 `/cv_splits` (80%-20% train-test) as evaluated in Fig. 3b.

We will create a different conda environment to run LiON. As introduced in our prior [repo](https://github.com/jswitten/LNP_ML), we will create conda environment `lnp_ml` as follows.

```
conda create -n lnp_ml python=3.8
conda activate lnp_ml
pip install chemprop==1.7.0
```

To train LiON on LNPDB for a single split (70%-15%-15% train-validation-test), run the following command in terminal. Note that the trained model checkpoints are already provided in this repository, so it is unnecessary to run the following command.

```
chemprop_train \
  --dataset_type regression \
  --data_path data/LNPDB_for_LiON/single_split/train.csv \
  --features_path data/LNPDB_for_LiON/single_split/train_extra_x.csv \
  --data_weights_path data/LNPDB_for_LiON/single_split/train_weights.csv \
  --separate_val_path data/LNPDB_for_LiON/single_split/val.csv \
  --separate_val_features_path data/LNPDB_for_LiON/single_split/val_extra_x.csv \
  --separate_test_path data/LNPDB_for_LiON/single_split/test.csv \
  --separate_test_features_path data/LNPDB_for_LiON/single_split/test_extra_x.csv \
  --save_dir data/LNPDB_for_LiON/single_split/trained_model_checkpoints \
  --target_columns Experiment_value
```

To use the trained LiON model to predict the test data for this single split, run the following command. Again, the test results dataframe is already provided in this repository, so it is unnecessary to run the following command.

```
chemprop_predict 
  --checkpoint_dir data/LNPDB_for_LiON/single_split/trained_model_checkpoints \
  --test_path data/LNPDB_for_LiON/single_split/test.csv \
  --features_path data/LNPDB_for_LiON/single_split/train_extra_x.csv \
  --preds_path data/LNPDB_for_LiON/single_split/test_results.csv
```

Moreover, fingerprint can be extracted from the penultimate linear layer of the LiON model’s feedforward neural network using the following command. 

```
chemprop_fingerprint 
  --checkpoint_dir data/LNPDB_for_LiON/single_split/trained_model_checkpoints \
  --test_path data/LNPDB_for_LiON/all_data.csv \
  --features_path data/LNPDB_for_LiON/all_data_extra_x.csv \
  --preds_path data/LNPDB_for_LiON/single_split/fingerprints_all_data.csv 
```

To train & test LiON on LNPDB for for 5 cross-validation splits with a particular heldout dataset (e.g., LM_2019), run the following command in terminal. Note that the trained model checkpoints and test results dataframe are already provided in this repository, so it is unnecessary to run the following commands.

```
for i in {0..4}
do
  echo "Training model for cv_$i"
  chemprop_train \
    --dataset_type regression \
    --data_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/train.csv \
    --features_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/train_extra_x.csv \
    --data_weights_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/train_weights.csv \
    --separate_val_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/test.csv \
    --separate_val_features_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/test_extra_x.csv \
    --separate_test_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/test.csv \
    --separate_test_features_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/test_extra_x.csv \
    --save_dir data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/trained_model_checkpoints \
    --target_columns Experiment_value
    
  echo "Running prediction for cv_$i"
  chemprop_predict \
    --checkpoint_dir data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/trained_model_checkpoints \
    --test_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/heldout_data.csv \
    --features_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/heldout_data_extra_x.csv \
    --data_weights_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/heldout_data_weights.csv \
    --preds_path data/LNPDB_for_LiON/heldout/heldout_LM_2019/cv_splits/cv_$i/heldout_data_results.csv
done
```

## Training & testing AGILE deep learning model

The AGILE deep learning model ([Xu et al., *Nat Commun*, 2024](https://www.nature.com/articles/s41467-024-50619-z)) should be cloned from this [repo](https://github.com/bowang-lab/AGILE) into LNPDB as follows to properly predict delivery efficacy on LNPDB data.

```
cd LNPDB/data/LNPDB_for_AGILE
git clone https://github.com/bowang-lab/AGILE
cd AGILE
unzip data.zip -d data
```

We will create a different conda environment `agile` to run AGILE. These instructions follow what was given in the AGILE [repo](https://github.com/bowang-lab/AGILE).

```
conda create -n agile python=3.9
conda activate agile
```

Install necessary dependencies as follows.

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric==2.2.0 torch-sparse==0.6.16 torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install -r requirements.txt
pip install rdkit
pip install "numpy<1.23.0,>=1.16.5"
pip install --force-reinstall mordred
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-build-isolation --disable-pip-version-check .
```

Refer to the notebook `/scripts/LNPDB_AGILE_training.ipynb` to train and evaluate AGILE models. The notebook involves the following: (1) installing requirements, (2) splitting data into test/train splits, (3) finetuning models on cross-validation splits, (4) generating LNPDB data feature descriptors, (5) evaluating models on AGILE and LNPDB data.

## Protonate ionizable lipids

Refer to the notebook `/scripts/protonate_ionizable_lipids.ipynb` to protonate ionizable lipid SMILES in LNPDB.

______

# Citation

**LNPDB: Lipid Nanoparticle Database towards structure-function modeling and data-driven design for nucleic acid delivery**

Evan Collins\*, Jungyong Ji\*, Sung-Gwang Kim\*, Jacob Witten, Seonghoon Kim, Richard Zhu, Peter Park, Minjun Jung, Aron Park, Rajith S. Manan, Arnab Rudra, Gyochang Keum, Eun-Kyoung Bang, Jun-O Jin, William J. Jeang, Robert S. Langer, Daniel G. Anderson\*, Wonpil Im\*

*In revision*
