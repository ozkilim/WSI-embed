import numpy as np
import time
import os
import argparse
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import openslide
import tiffslide
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import cProfile
import pstats
import io

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, device, model_name, verbose=1):
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    model.eval()
    
    for count, data in enumerate(tqdm(loader)):
        try:
            if verbose > 0:
                print(f"Processing batch {count + 1}/{len(loader)}")
            
            with torch.no_grad():
                batch = data['img'].to(device, non_blocking=True)
                coords = data['coord'].numpy().astype(np.int32)
                features = model(batch)
                
                if model_name == "virchow":
                    class_token = features[:, 0]
                    patch_tokens = features[:, 1:]
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                elif model_name == "virchow_v2":
                    class_token = features[:, 0]
                    patch_tokens = features[:, 5:]
                    features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                else:
                    print(f"Features shape: {features.shape}")

                features = features.cpu().numpy()

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'

        except Exception as e:
            print(f"Error processing batch {count + 1}: {e}")
            raise
    
    return output_path

def process_slide(args, slide_id, bag_name, h5_file_path, slide_file_path, output_path, model, img_transforms):
    if not args.no_auto_skip and slide_id+'.pt' in args.dest_files:
        print(f'Skipped {slide_id}')
        return

    time_start = time.time()

    if args.slide_ext == ".tiff":
        wsi = tiffslide.open_slide(slide_file_path) 
    else:
        wsi = openslide.open_slide(slide_file_path) 

    dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=img_transforms)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **args.loader_kwargs)
        
    try:
        output_file_path = compute_w_loader(output_path, loader=loader, model=model, device=device, model_name=args.model_name, verbose=1)
    except Exception as e:
        print(f"Error processing {slide_id}: {e}")
        return

    time_elapsed = time.time() - time_start
    print(f'\nComputing features for {output_file_path} took {time_elapsed} s')

    with h5py.File(output_file_path, "r") as file:
        features = np.array(file['features'])
        print('Features size: ', features.shape)
        print('Coordinates size: ', file['coords'].shape)

    features = torch.from_numpy(features)
    bag_base, _ = os.path.splitext(bag_name)
    torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'), _use_new_zipfile_serialization=True)

    return slide_id

def main(args):
    print('Initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    args.dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model = nn.DataParallel(model).to(device)
    model.eval()

    total = len(bags_dataset)
    args.loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'prefetch_factor': 2} if device.type == "cuda" else {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for bag_candidate_idx in range(total):
            slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
            bag_name = slide_id + '.h5'
            h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
            slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
            output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
            
            futures.append(executor.submit(
                process_slide, args, slide_id, bag_name, h5_file_path, 
                slide_file_path, output_path, model, img_transforms
            ))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            processed_slide = future.result()
            if processed_slide:
                print(f'Completed processing for slide: {processed_slide}')

    print('Feature extraction completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, default=None)
    parser.add_argument('--data_slide_dir', type=str, default=None)
    parser.add_argument('--slide_ext', type=str, default= '.svs')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet50_trunc', 
                        choices=['resnet50_trunc', 'uni_v1', 'conch_v1','prov_giga_path','virchow','virchow_v2','H-optimus-0'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Set up profiler
    pr = cProfile.Profile()
    pr.enable()

    # Run the main function
    main(args)

    # Disable profiler and print stats
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()

    print('Profiling Results:')
    print(s.getvalue())

    # Optionally, save profiling results to a file
    with open('profiling_results.txt', 'w') as f:
        f.write(s.getvalue())