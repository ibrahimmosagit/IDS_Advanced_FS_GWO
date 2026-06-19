# IIoT Intrusion Detection Feature Selection Framework

This repository provides the experimental notebooks and preprocessing scripts used for multi-class intrusion detection in IIoT environments. The study evaluates a GWO-guided top-k feature-ranking framework and compares it with multiple baselines on `CIC IIoT Dataset 2025` and `Edge-IIoTset`.

## Repository Contents

- `Summary_Analysis_and_Preprocessing_cic.ipynb`  
  Preprocessing and summary analysis for `CIC IIoT Dataset 2025`.

- `preprocess_edge_iiotset.py`  
  Preprocessing script for `Edge-IIoTset`.

- `Optimizing_IDS_Advanced_FS_GWO_CIC_IIoT_Dataset_2025.ipynb`  
  Main experimental notebook for `CIC IIoT Dataset 2025`.

- `Optimizing_IDS_Advanced_FS_GWO_Edge_IIoTset.ipynb`  
  Main experimental notebook for `Edge-IIoTset`.

- `Dataset_Shift_Analysis.ipynb`  
  Structural analysis notebook used to compare the preprocessed feature spaces of `CIC IIoT Dataset 2025` and `Edge-IIoTset`, including class balance, entropy, correlation, and PCA-based analysis.

## Datasets

This repository assumes access to the following datasets:

- `CIC IIoT Dataset 2025`
- `Edge-IIoTset`

Because dataset redistribution may be restricted, the raw data may need to be obtained from their original sources before running the notebooks.

## Workflow

A typical workflow is:

1. Run `Summary_Analysis_and_Preprocessing_cic.ipynb` to preprocess and inspect `CIC IIoT Dataset 2025`.
2. Run `preprocess_edge_iiotset.py` to preprocess `Edge-IIoTset`.
3. Run `Optimizing_IDS_Advanced_FS_GWO_CIC_IIoT_Dataset_2025.ipynb` for the main experiments on `CIC IIoT Dataset 2025`.
4. Run `Optimizing_IDS_Advanced_FS_GWO_Edge_IIoTset.ipynb` for the main experiments on `Edge-IIoTset`.
5. Run `Dataset_Shift_Analysis.ipynb` to analyze structural differences between the two preprocessed datasets.

## Main Experimental Settings

The study uses the following general setup:

- multi-class intrusion detection
- top-k feature ranking
- repeated runs with different random seeds
- stratified cross-validation
- class-weighted learning
- evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-macro
  - MCC
- efficiency-oriented reporting using:
  - feature-selection time
  - training time
  - inference time
  - memory usage

## Notes

- Some notebooks may contain saved outputs from completed runs.
- Running the full optimization notebooks can be computationally expensive and may take a long time depending on hardware and dataset size.
- File paths inside notebooks may need to be updated to match your local environment.

## Reproducibility

To improve reproducibility:

- keep the same random seeds used in the notebooks
- preserve the same preprocessing pipeline
- use the same train/test and validation settings described in the notebooks
- install compatible package versions before execution
