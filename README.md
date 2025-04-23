# SAGP: Semi-supervised Adversarial Graph Prediction for RUL Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of the paper ["Graph-based Semi-supervised Multi-objective Adversarial Learning for Complex System Remaining Useful Life Prediction"](https://example.com).

## Key Features
- **Graph-based Representation**: Models sensor correlations via KNN/Radius graphs
- **Semi-supervised Learning**: Jointly utilizes failure & suspension histories
- **Adversarial Training**: Enhances robustness with node/edge perturbations
- **Uncertainty Quantification**: Bayesian layers for prediction confidence
- **Dynamic Weight Adjustment**: Automatically balances loss components

# Implementation of SAGP.

Publication:
Title: Graph-based Semi-supervised Multi-objective Adversarial Learning for Complex System Remaining Useful Life Prediction



## Installation
```bash
git clone https://github.com/vijio/SAG_Prediction
cd SAGP-RUL-Prediction
pip install -r requirements.txt