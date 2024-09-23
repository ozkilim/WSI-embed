import numpy as np
import time
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
import tiffslide
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, device, model_name, verbose=1):
    """
    Args:
        output_path: directory to save computed features (.h5 file)
        loader: DataLoader object
        model: pytorch model
        device: torch device
        verbose: level of feedback
    """
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    model.eval()  # Ensure the model is in evaluation mode

    for count, data in enumerate(tqdm(loader)):
        try:
            if verbose > 0:
                print(f"Processing batch {count + 1}/{len(loader)}")

            with torch.inference_mode(), autocast():  # Use mixed precision inference
                batch = data['img']
                coords = data['coord'].numpy().astype(np.int32)

                batch = batch.to(device, non_blocking=True)  # Use non-blocking transfer
                features = model(batch)

                if model_name == "virchow":
                    class_token = features[:, 0]    # size: n x 1280
                    patch_tokens = features[:, 1:]  # size: n x 256 x 1280
                    # concatenate class token and average pool of patch tokens
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: n x 2560

                elif model_name == "virchow_v2":
                    class_token = features[:, 0]    # size: 1 x 1280
                    patch_tokens = features[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

                    # concatenate class token and average pool of patch tokens
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

                else:
                    print(f"Features shape: {features.shape}")

                features = features.cpu().numpy().astype(np.float32)

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'  # Append mode for subsequent saves

        except Exception as e:
            print(f"Error processing batch {count + 1}: {e}")
            raise  # Re-raise the exception to stop the loop if necessary

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'prov_giga_path', 'virchow', 'virchow_v2', 'H-optimus-0'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':

    print('Initializing dataset...')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
            
    model = model.eval()

    print("Initializing process group")

    dist.init_process_group(backend='nccl')
    print("Process group initialized")

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Debugging info about ranks and world size
    print(f"Rank: {local_rank}/{world_size}")

    # Split the dataset across processes
    total_files = len(bags_dataset)
    files_per_process = total_files // world_size
    remainder = total_files % world_size

    # Each process gets an even number of files, and the remainder is distributed
    if local_rank < remainder:
        start_idx = local_rank * (files_per_process + 1)
        end_idx = start_idx + (files_per_process + 1)
    else:
        start_idx = local_rank * files_per_process + remainder
        end_idx = start_idx + files_per_process

    # Each process works on its own subset of the dataset
    subset_dataset = bags_dataset[start_idx:end_idx]
    print(f"Process {local_rank} is handling files from {start_idx} to {end_idx}, total: {len(subset_dataset)}")

    # If there are no files assigned to the current process
    if len(subset_dataset) == 0:
        print(f"Process {local_rank} has no files to process, exiting.")
        exit(0)

    # Set the device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"Using device: {device}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    print(f"Model moved to device for rank {local_rank}")

    loader_kwargs = {'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(len(subset_dataset))):
        
        slide_id = subset_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        

        print(f"Processing slide: {slide_id}")

        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print(f'\nProgress: {bag_candidate_idx}/{len(subset_dataset)}')

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print(f'Skipped {slide_id}')
            continue 

        output_path = os.path.join(args.feat_dir, 'h5_files', f"{bag_name}.h5")  # No unique file per process, just use the same name
        time_start = time.time()

        # Load the slide
        if args.slide_ext == ".tiff":
            wsi = tiffslide.open_slide(slide_file_path)
        else:
            print(slide_file_path)
            wsi = openslide.open_slide(slide_file_path)

        # Create the dataset and dataloader
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                    wsi=wsi, 
                                    img_transforms=img_transforms)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        
        # Compute features for the assigned data
        try:
            output_file_path = compute_w_loader(output_path, loader=loader, model=model, device=device, model_name=args.model_name, verbose=1)
        except Exception as e:
            print("Error processing slide")
            print(e)

        time_elapsed = time.time() - time_start
        print(f'Computing features for {output_file_path} took {time_elapsed:.2f} s')

        # Optionally save the results
        with h5py.File(output_file_path, "r") as file:
            features = file['features'][:]
            print(f'Features size: {features.shape}')
            print(f'Coordinates size: {file["coords"].shape}')

        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
