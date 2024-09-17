#!/bin/bash

# Function to run the CLAM pipeline (patching and embedding)
run_clam_pipeline() {
  local raw_slides_dir=$1
  local patch_save_dir=$2
  local csv_path=$3
  local slide_ext=$4
  local feat_dir=$5

  echo "Running pipeline for slides in: $raw_slides_dir"

  # Patching
  CUDA_VISIBLE_DEVICES=$cuda_devices_patch python CLAM/create_patches_fp.py \
    --source "$raw_slides_dir" \
    --save_dir "$patch_save_dir" \
    --patch_size "$patch_size" \
    --step_size "$step_size" \
    --preset "$preset" \
    --seg \
    --patch \
    --stitch \
    --patch_level "$patch_level" \
    --max_patches "$max_patches"

  # Embedding
  CUDA_VISIBLE_DEVICES=$cuda_devices_embed python CLAM/extract_features_fp.py \
    --data_h5_dir "$patch_save_dir" \
    --data_slide_dir "$raw_slides_dir" \
    --csv_path "$csv_path" \
    --slide_ext "$slide_ext" \
    --feat_dir "$feat_dir" \
    --model_name "$model" \
    --batch_size "$batch_size"

  echo "Completed pipeline for slides in: $raw_slides_dir"
}

# Function to read YAML file
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file.yml>"
    exit 1
fi

# Read configuration
eval $(parse_yaml "$1")

# Run pipeline for the dataset
run_clam_pipeline "$raw_slides_dir" "$patch_save_dir" "$csv_path" "$slide_ext" "$feat_dir"