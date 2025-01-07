# WSI Embed

This project provides a pipeline for processing Whole Slide Images (WSIs) through patching and embedding using various foundation models.

## Overview

WSI Embed automates the process of:
1. Patching: Dividing large WSIs into smaller, manageable patches.
2. Embedding: Generating feature embeddings for these patches using pre-trained foundation models.

## Features

- Supports multiple foundation models for embedding generation.
- Configurable parameters for patching and embedding processes.
- Flexible configuration through YAML files.
- Parallel processing capabilities for improved performance.


## Usage

To run the pipeline, use the following command:

conda activate clam

./embed.sh configs/config.yaml


To join features based on case_id to make patient level bags : 

./patient_bag.sh 


Example: 

python script_name.py --cohort_csv cohort.csv --base_path /path/to/slide/files --output_dir /path/to/output --overwrite
