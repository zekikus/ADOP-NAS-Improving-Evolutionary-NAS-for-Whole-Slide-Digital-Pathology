# ADOP-NAS-Improving-Evolutionary-NAS-for-Whole-Slide-Digital-Pathology
ADOP-NAS: Improving Evolutionary NAS for Whole-Slide Digital Pathology with Diverse Initialization and Adaptive Mutation

This repository contains the official PyTorch implementation of the paper **"ADOP-NAS: Improving Evolutionary NAS for Whole-Slide Digital Pathology with Diverse Initialization and Adaptive Mutation"**. 

The project focuses on applying an improved Evolutionary Neural Architecture Search (NAS) method to Whole Slide Images (WSI) in digital pathology. The framework is validated across four distinct datasets: **BCNB**, **EBHI**, **SPIDER**, and **TCGA**.

## Repository Structure

The repository is organized into four main directories, each corresponding to a specific dataset. Each directory contains the necessary scripts to perform architecture search, training, and testing.

```text
.
├── BCNB/
├── EBHI/
├── SPIDER/
└── TCGA/
```

## Core Scripts

Inside each dataset folder (e.g., TCGA/), you will find the following core scripts:

#### ode.py:

*   **Purpose:** Executes the Evolutionary Neural Architecture Search (NAS).
*   **Function:** This script runs the search algorithm described in the paper to discover the optimal neural network architecture for the specific dataset. It utilizes the proposed differential evolution strategy to navigate the search space efficiently.

#### train_pytorch.py:

*   **Purpose:** Retrains the best-found architectures.
*   **Function:** After the NAS phase (ode.py) identifies the best model architecture, this script is used to train that specific model from scratch for a longer duration (more epochs) to achieve maximum performance.

#### test_pytorch.py:

*   **Purpose:** Inference and Evaluation.
*   **Function:** This script loads the fully trained model weights and evaluates its performance on the held-out test dataset, reporting metrics such as Accuracy, F1-Score, etc.

### Usage

To use this framework, follow the steps below for any of the datasets (e.g., inside the BCNB folder).

#### 1. Architecture Search

Run the evolutionary search to find the optimal architecture:

```bash
cd BCNB
python ode.py
```

#### 2. Training the Best Model

Once the best architecture is found, train it fully:

```bash
python train_pytorch.py
```

#### 3. Testing

Evaluate the trained model on the test set:

```bash
python test_pytorch.py
```

## Reproducibility
You can download trained model weights and detailed prediction results for each dataset from FigShare:
* Pretrained weights for [TCGA and SPIDER](https://doi.org/10.6084/m9.figshare.31048645)
* Pretrained weights for [EBHI and BCNB](https://doi.org/10.6084/m9.figshare.31054714)
