# Latent Cluster VAE (LCVAE)

This repository contains the official implementation of the **Latent Cluster Variational Autoencoder (LCVAE)**, a novel generative model for producing high-quality synthetic tabular data in the context of network intrusion detection. LCVAE explicitly structures the latent space into class-specific ellipsoidal clusters, enabling interpretable and controllable sample generation with tunable diversity.

---

## 📁 Repository Structure

### Core Models
- `lcvae.py`: Implementation of the Latent Cluster VAE model, including cluster-aware latent space design and sampling control via the epsilon parameter.
- `gmvae.py`: Baseline Gaussian Mixture VAE model for comparison, using a learned mixture in latent space.

### Training Scripts
- `train_lcvae.ipynb`: Training notebook for the LCVAE model on NSL-KDD and CICIDS2017 datasets.
- `train_gmvae.ipynb`: Training notebook for the GMVAE model using the same experimental setup.

### Generation Scripts
- `generation_lcvae.ipynb`: Generates synthetic data using the trained LCVAE model, with configurable dispersion parameter \(\epsilon\).
- `generation_gmvae.ipynb`: Generates synthetic samples from the GMVAE baseline.
- `generation_other_models.ipynb`: Generation scripts for external models (TVAE, CTGAN, CopulaGAN, SMOTE, ADASYN) for comparative purposes.

### Evaluation
- `evaluation_metrics.ipynb`: Computes evaluation metrics such as fidelity (Wasserstein), diversity (Vendi, Diversity Gap), class balance, and utility (TSTR).
- `evaluation_graphics.ipynb`: Generates plots and visual comparisons of the results across models and datasets.
- `global_results_nslkdd.csv` / `global_results_cicids.csv`: Aggregated performance metrics for each generation method.

### Generations & Results
- `generations_nsl/`, `generations_cicids/`: Generated datasets from LCVAE, GMVAE, and external models on NSL-KDD and CICIDS2017.
- `results_nsl/`, `results_cicids/`: Evaluation outputs (figures, scores) for each dataset.

### Preprocessing & Feature Engineering
- `Preprocessing.ipynb`: Prepares the datasets (NSL-KDD, CICIDS2017), including label encoding, feature selection, and formatting.
- `typed_nslkdd_all_features.pkl` / `typed_cicids2017_all_features.pkl`: Pickled objects containing typed schema for each dataset, used in transformation pipelines.
- `typed_data_transformer_gmm.py`: Utility script for applying GMM-based preprocessing transformations (e.g., for continuous or integer features).

### Cluster Management
- `cluster_center_generation.py`: Generates class-wise latent cluster centers distributed on a hypersphere with minimal overlap, used to structure the LCVAE latent space.

---

## Reproducibility

All notebooks are designed to run independently with fixed seeds and follow the same experimental protocol for fair comparison. If using this repository for your work, please cite the original paper and our contribution accordingly.

---

## Acknowledgments

This work reuses preprocessing, baseline models, and evaluation pipelines from [AdenRajput/Comparative_Analysis](https://github.com/AdenRajput/Comparative_Analysis), and extends it with explicit latent structuring and controllable generation via LCVAE.
