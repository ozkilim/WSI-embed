import os
import argparse
import pandas as pd
import torch

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Concatenate slide-level .pt files into patient-level .pt files.")
    parser.add_argument("--cohort_csv", type=str, required=True, help="Path to the cohort CSV file.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the .pt files for slides.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for patient-level .pt files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files in the output directory.")
    args = parser.parse_args()

    # Load cohort CSV
    cohort_df = pd.read_csv(args.cohort_csv)
    if 'case_id' not in cohort_df.columns or 'slide_id' not in cohort_df.columns:
        raise ValueError("The cohort CSV must contain 'case_id' and 'slide_id' columns.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Group by case_id
    grouped = cohort_df.groupby('case_id')

    for case_id, group in grouped:
        output_path = os.path.join(args.output_dir, f"{case_id}.pt")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {case_id}.pt (already exists). Use --overwrite to overwrite.")
            continue

        # Collect and concatenate features for all slide_ids
        features_list = []
        for slide_id in group['slide_id']:
            slide_path = os.path.join(args.base_path, f"{slide_id}.pt")
            if not os.path.exists(slide_path):
                print(f"Warning: File {slide_path} not found, skipping.")
                continue

            features = torch.load(slide_path)
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"File {slide_path} does not contain a valid PyTorch tensor.")

            features_list.append(features)

        if not features_list:
            print(f"Warning: No valid features found for case_id {case_id}. Skipping.")
            continue

        # Concatenate features along the first dimension (assuming row-wise concatenation)
        patient_features = torch.cat(features_list, dim=0)

        # Save the concatenated features as case_id.pt
        torch.save(patient_features, output_path)
        print(f"Saved {output_path} with shape {patient_features.shape}.")

if __name__ == "__main__":
    main()
