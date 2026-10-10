# IFGEELMMV: Intuitionistic Fuzzy Graph-Embedded Extreme Learning Machine for Multi-View Learning

## Overview

This repository provides the MATLAB implementation of **IFGEELMMV**, an Intuitionistic Fuzzy Graph-Embedded Extreme Learning Machine designed for multi-view learning. The framework integrates intuitionistic fuzzy modeling, graph-based learning, and Extreme Learning Machine (ELM) principles to support classification using multi-view data.

The implementation includes model training, hyperparameter optimization through grid search, and comprehensive performance evaluation.

## Citation

If you use this code in your research, please cite the associated research paper:

**[Insert the complete paper citation here]**

If the paper has not yet been published, please update this section when the publication details become available.

## Experimental Environment

The experiments were conducted using MATLAB on a computing system with the following configuration:

- **MATLAB Version:** [Specify MATLAB version]
- **Processor:** [Specify CPU model]
- **Clock Speed:** [Specify processor clock speed]
- **RAM:** [Specify RAM capacity]
- **Operating System:** [Specify operating system]

## Repository Contents

- **`Part1_Main_Pipeline.m`** — Contains the main pipeline for loading datasets and initiating the experimental workflow.
- **`Part2_Model_Functions.m`** — Contains the model-related functions for intuitionistic fuzzy processing, graph/LFDA-related computations, and classification.
- **`Part3_Evaluation_and_Plots.m`** — Contains the evaluation procedures, hyperparameter search routines, and performance visualization functions.
- **`README.md`** — Provides an overview of the repository and instructions for using the code.

*Note: The MATLAB files are organized into separate parts for repository management. Their function dependencies and execution order should be checked before running them as independent files.*

## Dataset Format

Prepare the dataset in CSV format before running the experiments.

- Each CSV file should contain the input features and corresponding class labels.
- The **last column should contain the class labels**.
- Place the dataset files in the designated `datasets` folder.
- Ensure that the dataset format is consistent with the data-loading procedure implemented in the code.

## Hyperparameter Optimization

The implementation includes a grid-search procedure for evaluating model configurations. The hyperparameter ranges and experimental settings should be configured according to the corresponding research paper or the settings specified in the code.

## Performance Evaluation

The implementation supports the evaluation of classification performance using the following metrics, where implemented by the code:

- Area Under the ROC Curve (AUC)
- Geometric Mean (G-mean)


The experimental results can be saved for subsequent analysis and comparison.

## How to Use

1. Clone or download this repository.
2. Place the required CSV datasets in the `datasets` folder.
3. Open MATLAB and navigate to the repository directory.
4. Check the function dependencies and execution order of the MATLAB files.
5. Run the appropriate main script and configure the dataset and hyperparameters as required.
6. Review the generated performance metrics and experimental outputs.

## Issues and Contact

If you encounter any bugs, errors, or implementation-related issues, please open an issue in this GitHub repository.

For research-related queries, contact:

**Author:** [Your name]  
**Email:** [Your email address]

## Disclaimer

This repository is intended for research and academic use. Please refer to the associated research paper for the complete methodology, theoretical formulation, and detailed experimental settings.
