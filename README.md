# IFGEELMMV: Intuitionistic Fuzzy Graph-Embedded Extreme Learning Machine for Multi-View Learning

Please cite the following paper if you are using this code.

**Reference:** [Author 1], and [Author 2] ([Year]). "[Full paper title: IFGEELMMV ...]", [Journal / Conference name], [Publisher] ([Status, e.g., Under Review / In Revision / Published]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The experiments are executed on a computing system possessing MATLAB [version, e.g., R2024a] with the Statistics and Machine Learning Toolbox and (optionally) the Parallel Computing Toolbox, an [CPU model] processor operating at [clock speed] with [RAM size] of Random Access Memory (RAM), and a [Operating system] operating platform.

We have put a demo of the IFGEELMMV model with the "[dataset name]" dataset.

In this demonstration, the code is run with a grid search over the hyperparameters for the "[dataset name]" dataset. For the detailed hyperparameter setting, please refer to the paper.

## Description of files

- **`IFGWELM_MV_ablation.m`**: This is the main file. It runs the IFGEELMMV model and its ablation variants on every dataset present in the `datasets` folder. It contains the dataset loader, the intuitionistic fuzzy weighting, the graph (LFDA) regularization, the training function, the 5-fold grid search and the evaluation metrics. No extra files are needed.
- **`datasets/`**: Folder that contains the datasets (`.csv` files).
- **`cumulative_ablation_IFGWELM_MV.csv`**: Output file generated after running the code.

## Folder structure

```
IFGEELMMV/
    IFGWELM_MV_ablation.m
    datasets/
        your_dataset.csv
```

The `datasets` folder must be in the same folder as `IFGWELM_MV_ablation.m`. The code finds the folder automatically, so no path needs to be edited.

## Dataset format

- Each dataset is a `.csv` file with a header row.
- The **last column** is the class label and must contain exactly **two classes** (for example 0/1, -1/+1 or text labels).
- All other columns are features (at least two). Categorical features are label-encoded automatically.
- Rows containing `NaN` or `Inf` are removed.
- Features are split into two halves to form the two views.

## How to run

1. Place `IFGWELM_MV_ablation.m` and the `datasets` folder in the same directory.
2. Open `IFGWELM_MV_ablation.m` in MATLAB.
3. Click **Run** (or type `IFGWELM_MV_ablation` in the Command Window from that folder).

Do not call the internal functions (such as `train_joint_multiview_fuzzy_welm`) directly from the Command Window, because they need inputs that the main script creates.

## Hyperparameters

The hyperparameters are tuned once on the full model with a 5-fold cross-validated grid search (selection by G-mean). The same best values are then used for all ablation variants.

| Parameter | Description | Search range |
|-----------|-------------|--------------|
| `h` | Number of hidden nodes | {30, 60, 100, 150} |
| `C1`, `C2` | Weights of the data-fitting terms of view 1 and view 2 | 2^{-5, 0, 5, 10, 15} |
| `rho` | Multi-view coupling parameter | {0.001, 0.1, 1} |
| `lambda` (`theta` in the code) | Graph regularization parameter (`lambda1 = lambda2`) | {0.001, 0.1, 1} |
| `mew` | Kernel width of the intuitionistic fuzzy weighting | 2^{-10, 0, 10} |

Total number of combinations: 2700 per dataset.

## Ablation variants

| Variant | Description |
|---------|-------------|
| `Full_IFGWELM_MV` | Complete proposed model |
| `w_o_Fuzzy` | Intuitionistic fuzzy weighting removed (all weights equal to 1) |
| `w_o_LFDA` | Graph (LFDA) regularization removed |
| `w_o_MultiView` | Multi-view coupling removed (`rho = 0`) |
| `w_o_RobustScaling` | Robust scaling replaced by z-score scaling |

## Output

The file `cumulative_ablation_IFGWELM_MV.csv` is created in the same folder as the script. It has the columns:

`Dataset, Variant, Accuracy, F1, AUC, Gmean, Sensitivity, Recall, Specificity, Precision, MCC, Time`

All values are averages over the 5 folds. Time is the average training and testing time per fold in seconds.

## Requirements

- MATLAB [version] or later
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (optional; the code runs serially if it is not available)

## Contact

If you find any bugs/issues, please write to [Author name] ([email address]).
