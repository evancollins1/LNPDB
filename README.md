# LNPDB: structure-function database of lipid nanoparticles to advance data-driven design for nucleic acid delivery

<p align="center">
  <img src="data/misc/LNPDB_fig1.png" alt="alt text" width="800px" align="middle"/>
</p>

*Figure: LNPDB (https://lnpdb.molcube.com) is a data repository and web tool for compiling and uploading LNP structure-function data.*

## Description

This repository contains processing code to analyze the molecular dynamics trajectories evaluated in the paper "LNPDB: structure-function database of lipid nanoparticles to advance data-driven design for nucleic acid delivery" (*under review*).

_____

# Table of contents

- [Citation](#citation)
- [Repository structure](#repository-structure)
- [System requirements](#system-requirements)
- [Installation guide](#installation-guide)

_____

# Citation

LNPDB: structure-function database of lipid nanoparticles to advance data-driven design for nucleic acid delivery. Evan Collins\*, Jungyong Ji\*, Sung-Gwang Kim\*, Jacob Witten, Seonghoon Kim, Richard Zhu, Peter Park, Minjun Jung, Aron Park, Rajith S. Manan, Arnab Rudra, Gyochang Keum, Eun-Kyoung Bang, Jun-O Jin, Robert S. Langer, Daniel G. Anderson\*, Wonpil Im\*

*Under review.*

# Repository structure

The repository is structured as follows:

- `/data`: contains molecular dynamics trajectory data. Two example trajectories from the paper are provided.

- `/results`: contains the results for the different measures derived from molecular dynamics trajectories.

- `/scripts`: contains the Python scripts used to process data and generate results.

# System requirements

## Hardware requirements

To run the analysis code in this repository, it is recommended to have a computer with enough RAM (> 8 GB) to support the in-memory operations. Moreover, robust GPUs are recommended to run the all-atom simulations.

## Software requirements

This code has been implemented with Python version 3.9.20. Other Python versions will likely work. See below for creating an appropriate conda environment.

### OS requirements

This code has been tested on the following systems, although it should work generally for other OS types (e.g., Windows) with potentially minor required changes to package versions.

- macOS: Ventura 13.2.1

- Linux: Ubuntu 22.04

# Installation guide

## Install repository

```
$ git clone https://github.com/evancollins1/LNPDB.git
```

## Install Python modules

Create a conda environment (`lnpdb`) with the modules specified in `environment.yml`.

```
$ conda env create -f environment.yml
$ conda activate lnpdb
```
