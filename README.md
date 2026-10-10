# IFGEELMMV: Intuitionistic Fuzzy-Enhanced Multi-View Graph-Embedded Extreme Learning Machine for Noisy and Imbalanced Classification

## Overview

This repository provides the MATLAB implementation of **IFGEELMMV** (Intuitionistic Fuzzy-Enhanced Multi-View Graph-Embedded Extreme Learning Machine for Noisy and Imbalanced Classification). The framework integrates intuitionistic fuzzy modeling, graph-based learning (fuzzy-weighted LFDA), and Extreme Learning Machine (ELM) principles to achieve robust binary classification using multi-view data.

The implementation includes an automated pipeline for dataset loading, feature scaling, grid-search hyperparameter optimization with 5-fold cross-validation, and comprehensive performance evaluation.

---

## Repository Contents

- **`IFGEELMMV.m`** — Main entry script. Handles dataset discovery, 5-fold grid search, evaluation, and saves results.
- **`functions/ifelm_multiview_train.m`** — Core training and testing  for the intuitionistic fuzzy graph-embedded multi-view ELM.
- **`functions/graph_embedding.m`** — Computes fuzzy-weighted graph regularization matrices.
- **`functions/evaluation_metrics.m`** — Evaluates performance metrics including G-mean and Area Under the ROC Curve (AUC).
- **`datasets/`** — Directory containing the input CSV datasets.


---

## Dataset Format

1. Prepare your benchmark datasets in **CSV format** and place them inside the `datasets/` folder.
2. Each CSV file should contain feature columns followed by the class label in the **last column**.
3. The labels must be binary classes (e.g., `{0, 1}` or `{-1, 1}`).
4. The feature space is automatically split into two halves internally to represent the two distinct views.

---

## Experimental Environment

- **Language:** MATLAB
- **Toolboxes Required:** Statistics and Machine Learning Toolbox
- **Optional Toolboxes:** Parallel Computing Toolbox (automatically utilizes `parpool` if available for accelerated grid search; falls back to serial execution otherwise).

---

## Hyperparameter Optimization

The model executes a grid search over key architectural and regularization parameters using 5-fold cross-validation, optimized primarily for **G-mean**:
- Number of hidden nodes ($h$)
- Regularization parameters ($C_1, C_2$)
- Multi-view coupling parameter ($\rho$)
- Graph regularization parameter ($\theta$)

---

## How to Run

1. Clone or download this repository to your local machine.
2. Ensure your CSV dataset files are placed inside the `datasets/` folder.
3. Open **MATLAB** and set the current folder to the root directory of this repository.
4. Open and run **`IFGEELMMV.m`** (or type `IFGEELMMV` in the MATLAB Command Window).
5. Once execution finishes, a summary table containing best hyperparameters and average metrics (G-mean, AUC) will be displayed and saved to **`IFGEELMMV_results.csv`**.

---

## Citation

If you use this code or benchmark findings in your research, please cite the corresponding paper

