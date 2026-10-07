# Master's Thesis (TU Delft - DSAIT Programme): DrugZip, CCSynergy, ChemCPA, and Sensitivity DNN

This repository contains the code for my thesis work at TU Delft. It provides the implementation and scripts necessary to generate DrugZip representations from a database of molecules, reproduce existing models (CCSynergy and ChemCPA), and train my custom Sensitivity Deep Neural Network (DNN).

The full thesis can be found at the TU Delft Education Repository:
[https://repository.tudelft.nl/record/uuid:10f9f08b-3059-4f3b-96ef-8b556a1b8404](https://repository.tudelft.nl/record/uuid:10f9f08b-3059-4f3b-96ef-8b556a1b8404)

## Table of Contents
- [Overview](#overview)
- [Data](#data)
<!--  - [Repository Structure](#repository-structure) -->
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [1. Generating DrugZip](#1-generating-drugzip)
  - [2. Reproducing CCSynergy](#2-reproducing-ccsynergy)
  - [3. Reproducing ChemCPA](#3-reproducing-chemcpa)
  - [4. Sensitivity DNN](#4-sensitivity-dnn)
- [Citation](#citation)
- [License](#license)

## Overview

The primary objectives of this codebase are:
1. **DrugZip Generation**: Processing a provided database of molecules to generate DrugZip embeddings/features.
2. **CCSynergy Reproduction**: Replicating the CCSynergy architecture for baseline comparisons.
3. **ChemCPA Reproduction**: Implementing the ChemCPA framework.
4. **Sensitivity DNN**: Training and evaluating a custom Deep Neural Network designed for sensitivity predictions.

## Data

The signature data is available on the official Chemical Checker website: <https://chemicalchecker.com/downloads/signature3>.
You may download each signature, file by file, and then use the csv_signature_generator.py to transalte them into usable csv file for the DrugZip compression.
<!-- 
## Repository Structure

* `data/` - Directory for storing the raw molecule database and generated outputs.
* `drugzip/` - Core logic for parsing molecules and generating DrugZip features.
* `models/` - Implementations of CCSynergy, ChemCPA, and the Sensitivity DNN.
* `scripts/` - Execution scripts for training, testing, and generating data.
-->

## Environment Setup

To ensure strict reproducibility (particularly for the CCSynergy and DNN reproduction), please use the exact environment specifications provided below. 

The setup relies on `conda` and `pip` to manage dependencies accurately. Open your terminal and execute the following:

#### 1. Create and Activate the Conda Environment

```bash
conda create -n ccsynergy -c conda-forge python=3.10 numpy=1.23.5 scipy=1.10.1 pandas=2.0.3 scikit-learn=1.3.0
conda activate ccsynergy
