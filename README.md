# Predicting Genome Completeness from UHGG Metadata

This project builds and compares machine learning models to predict genome **completeness** using metadata from the **Unified Human Gastrointestinal Genome (UHGG)** collection.

Genome completeness is a critical quality measure for metagenome-assembled genomes (MAGs), and is typically estimated using marker gene presence (e.g., via CheckM). In this project, we attempt to predict completeness based on features derived from genome assemblies such as contig counts, GC content, and rRNA/tRNA annotations.
While this is a pretty simple example, it can be useful to fill in missing completeness scores in large datasets.

---

## 🧠 Objective

Given metadata features for each genome:

- `Length`
- `N_contigs`
- `N50`
- `GC_content`
- `Contamination`
- `rRNA_5S`, `rRNA_16S`, `rRNA_23S`
- `tRNAs`

predict the `Completeness` score using:
- **Linear Regression**
- **Random Forest**
- **Neural Network**

---

## 🛠️ How to Run

### 1. Install Requirements

Make sure you have Python ≥ 3.8 and install the following packages:

- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

### 2. Prepare the Metadata

Ensure you have the UHGG metadata in a tab-separated format named `uhgg_metadata.tsv` in the same directory as the script. It should contain the following columns:

- `Length`, `N_contigs`, `N50`, `GC_content`, `Completeness`, `Contamination`, `rRNA_5S`, `rRNA_16S`, `rRNA_23S`, `tRNAs`

You can extract these from the full UHGG metadata using `pandas`.

### 3. Run the Script

Run the script `predict_completeness.py` from your Python environment. It will:

- Load and clean the dataset
- Train three models
- Evaluate performance
- Generate a plot comparing predicted vs. true values

### 4. Flags
You can specify the following flags when running the script:
- `--cv`: Whether to run in cross-validation mode (default: False)
- `--folds`: Number of folds for cross-validation (default: 5)
- `--epochs`: Number of epochs for the neural network (default: 100)
- `--dropout`: Dropout rate for the neural network (default: 0.2)

---

## 📈 Model Evaluation

Each model is evaluated using:
- **Mean Squared Error (MSE)**

The script also generates a scatterplot of predicted vs. actual completeness scores for all models.

---

## 🔬 Biological Relevance

Genome completeness is essential for:
- Filtering low-quality assemblies
- Ensuring downstream functional analysis is accurate
- Benchmarking pipelines and assemblies

By learning how genome features like `N50`, `GC_content`, and rRNA content relate to completeness, this project provides a lightweight predictive tool for assessing MAG quality.

---

## 📌 Potential Extensions in Progress

- Add SHAP values to interpret feature contributions
- Implement hyperparameter tuning for better model performance
- Try additional models (e.g., XGBoost, Ridge, SVR)
- Add network architecture visualization for the neural network
- Add better CPU/GPU support for TensorFlow

---

## 📄 License

MIT License — use freely with attribution.

---

## ✍️ Author

Nimrod Shai
