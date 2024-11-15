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
    --microns_per_patch_edge "$microns_per_patch_edge" \
    --preset "$preset" \
    --seg \
    --patch \
    --stitch \
    --patch_level "$patch_level" \
    # --process_list "$csv_path" \
    --max_patches "$max_patches"

  # Embedding
  CUDA_VISIBLE_DEVICES=$cuda_devices_patch python CLAM/extract_features_fp.py \
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


export OMP_NUM_THREADS=1





# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file.yml>"
    exit 1
fi

# Read configuration
eval $(parse_yaml "$1")




# TODO: somewthing that chacks the magnification before running and offer cli to pick ... or something ....
# Load the first slide from the raw slides directory and find its magnification

# Get the first slide path using your method
# first_slide_path=$(ls "$raw_slides_dir" | head -1 2>/dev/null)
# first_slide_path="$raw_slides_dir/$first_slide_path"
# echo "First slide path: $first_slide_path"

# # Call the Python script to get the magnification
# slide_magnification=$(python3 get_magnification.py "$first_slide_path" 2>&1)

# # Check if magnification was obtained
# if [[ "$slide_magnification" != "Unknown" && "$slide_magnification" != Error* ]]; then
#     echo "Successfully loaded the file and obtained the magnification."
# else
#     echo "Failed to load the file or obtain the magnification."
#     echo "Error details: $slide_magnification"
# fi

# echo "Magnification of the first slide: $slide_magnification"

# Run pipeline for the dataset
run_clam_pipeline "$raw_slides_dir" "$patch_save_dir" "$csv_path" "$slide_ext" "$feat_dir"