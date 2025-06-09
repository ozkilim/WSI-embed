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

# === MARKING REMOVAL CONFIGURATION ===

# Enable/disable colored marking removal during tissue segmentation (default: false)
remove_markings: false

# Optional: disable specific colors when remove_markings is true
# By default, all colors (red, blue, green, black, off_white) are removed when remove_markings is enabled
# Uncomment and set to true to disable specific color removal:
# disable_red_removal: false
# disable_blue_removal: false  
# disable_green_removal: false
# disable_black_removal: false
# disable_off_white_removal: false

# Advanced: Custom HSV color ranges (optional - uses sensible defaults if not specified)
# marking_colors:
#   red:
#     enabled: true
#     lower1: [0, 100, 80]    # First red range (low hue)
#     upper1: [10, 255, 255]
#     lower2: [170, 100, 80]  # Second red range (high hue)
#     upper2: [180, 255, 255]
#   blue:
#     enabled: true
#     lower: [100, 100, 80]
#     upper: [130, 255, 255]
#   green:
#     enabled: true
#     lower: [35, 5, 5]
#     upper: [90, 255, 255]
#   off_white:
#     enabled: true
#     lower: [0, 0, 200]
#     upper: [179, 30, 255]
#   black:
#     enabled: true
#     lower: [0, 0, 0]
#     upper: [179, 255, 80]

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

## 🎨 Marking Removal Feature

The pipeline now supports automatic removal of colored markings (pen marks, annotations, artifacts) from WSI tissue during segmentation. This is useful for slides that contain:

- **Red/blue pen markings** from pathologists
- **Green markings** from digitization artifacts  
- **Black markings** from various sources
- **Off-white patches** that may interfere with tissue detection

### Basic Usage

Simply enable marking removal in your config file:

```yaml
remove_markings: true
```

By default, this will remove all 5 color types (red, blue, green, black, off-white) using pre-configured HSV color ranges.

### Advanced Configuration

**Disable specific colors:**
```yaml
remove_markings: true
disable_green_removal: true  # Keep green markings, remove others
disable_red_removal: true    # Keep red markings, remove others
```

**Custom color ranges:**
```yaml
remove_markings: true
marking_colors:
  red:
    enabled: true
    lower1: [0, 120, 100]     # Adjust HSV ranges as needed
    upper1: [8, 255, 255]
    lower2: [172, 120, 100]
    upper2: [180, 255, 255]
  # ... other colors
```

### Command Line Usage

You can also control marking removal via command line arguments:

```bash
python CLAM/create_patches_fp.py \
  --source /path/to/slides \
  --save_dir /path/to/output \
  --remove_markings \
  --disable_green_removal \
  --seg --patch
```

### How It Works

1. **HSV Color Space**: Converts image to HSV for robust color detection
2. **Multi-Range Detection**: Red markings use two HSV ranges to handle hue wraparound
3. **Morphological Processing**: Applies dilation to ensure complete removal
4. **White Replacement**: Detected marking regions are set to white before tissue segmentation

---

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

 - ✅ ~~Add control over marker removal~~ (COMPLETED)
 - Add support for ensembling multiple models (e.g., VirchowV2 + Conch) as demonstrated in
 - Add batch processing optimization for very large cohorts
 - Add visualization tools for marking removal results
