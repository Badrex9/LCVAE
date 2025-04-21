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
- `generations_nsl/`: Contains a ZIP archive (`generations_nsl.zip`) that includes all generated datasets (LCVAE, GMVAE, and baselines) for the NSL-KDD dataset.
- `generations_cicids/`: Empty placeholder folder. The CICIDS2017 generated datasets were too large to include, even in compressed form. See the note below for access.
- `results_nsl/`, `results_cicids/`: Evaluation outputs (figures, metrics) for each dataset and generation method.

### Preprocessing & Feature Engineering
- `Preprocessing.ipynb`: Prepares the datasets (NSL-KDD, CICIDS2017), including label encoding, feature selection, and formatting.
- `typed_nslkdd_all_features.pkl` / `typed_cicids2017_all_features.pkl`: Pickled objects containing typed schema for each dataset, used in transformation pipelines.
- `typed_data_transformer_gmm.py`: Utility script for applying GMM-based preprocessing transformations (e.g., for continuous or integer features).

### Cluster Management
- `cluster_center_generation.py`: Generates class-wise latent cluster centers distributed on a hypersphere with minimal overlap, used to structure the LCVAE latent space.

---

## 📦 Data Availability

Due to GitHub's storage and bandwidth limitations, large files have been excluded from version control. Instead:

- The `dataset/` folder contains a ZIP archive with both NSL-KDD and CICIDS2017 preprocessed datasets.
- The `generations_nsl/` folder includes a ZIP archive containing all synthetic datasets generated from the NSL-KDD experiments.
- The `generations_cicids/` folder is left empty in the repository. Generated data for CICIDS2017 could not be included due to size constraints (even when compressed). Please use the provided generation scripts to reproduce the data, or contact the authors for access.

---

## 🔁 Reproducibility

All notebooks are designed to run independently with fixed random seeds and follow a shared experimental protocol for consistent benchmarking across models.

---

## 📝 Acknowledgments

This work reuses preprocessing steps, baseline models, and evaluation metrics from [AdenRajput/Comparative_Analysis](https://github.com/AdenRajput/Comparative_Analysis), and extends them with explicit latent structuring and a controllable generation mechanism via LCVAE.
