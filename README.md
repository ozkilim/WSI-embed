# Slide Crush: Fast WSI embedding.


![Alt text](./figures/slide_crush.webp)

This project provides a pipeline for processing Whole Slide Images (WSIs) through patching and embedding using various foundation models.

---

## 🧭 Overview

**WSI Embed** automates the process of:

1. **Patching** – Dividing large WSIs into smaller, manageable patches.
2. **Embedding** – Generating feature embeddings for these patches using pre-trained foundation models.

---

## ✨ Features

- ✅ Supports multiple foundation models for embedding generation
- ⚙️ Configurable parameters for patching and embedding
- 📄 YAML-based configuration files
- 🚀 Parallel processing for performance boost

---

## 🚀 Usage

### Activate the environment and run the pipeline with:

```bash
conda activate clam
```


### 📄 Create a config file: `configs/config.yaml`

```yaml
# === PATCHING PARAMETERS ===

# The physical size of one patch edge in microns (typically matches model training size)
microns_per_patch_edge: 128

# The resolution level of the WSI to use for patching (0 = highest resolution)
patch_level: 0

# File extension for WSIs (e.g., ".svs", ".tiff", ".mrxs")
slide_ext: ".svs"

# === MODEL CONFIGURATION ===

# Name of the model used for generating embeddings (must be supported by the pipeline)
model: "uni_v2"

# === INPUT / OUTPUT PATHS ===

# Directory where raw WSIs are stored
raw_slides_dir: "/tank/WSI_data/lung/TCGA-LUAD/slides"

# Directory where extracted patches will be saved
patch_save_dir: "/tank/WSI_data/lung/TCGA-LUAD/CLAM/20X/patches"

# Path to cohort CSV containing metadata (e.g., slide IDs, labels)
csv_path: "/mnt/ncshare/ozkilim/Histopathology/benchmarking/cohorts/LUAD_ralapse_TCGA.csv"

# Directory where the final feature embeddings will be saved
feat_dir: "/tank/WSI_data/lung/TCGA-LUAD/CLAM/20X/uni_v2"

# === PERFORMANCE / COMPUTE SETTINGS ===

# Number of patches to embed in one batch (depends on GPU memory)
batch_size: 512

# Safety cap on total number of patches processed (use a large number to avoid accidental limits)
max_patches: 3000000000

# Which CUDA device to use for patch extraction
cuda_devices_patch: "0"

# === OPTIONAL: CLAM PRESET ===

# Path to a CSV file containing preset hyperparameters or class definitions for CLAM models
preset: "CLAM/presets/bwh_biopsy.csv"

### Generate the embeddings with.

```bash
./embed.sh configs/config.yaml
```


### To join features based on case_id and create patient-level bags:

```bash
./patient_bag.sh
```

```bash
python patient_bag.py \
  --cohort_csv cohort.csv \
  --base_path /path/to/slide/files \
  --output_dir /path/to/output \
  --overwrite
```

## TODO:

 - Add control over marker removal
 - Add support for ensembling multiple models (e.g., VirchowV2 + Conch) as demonstrated in
