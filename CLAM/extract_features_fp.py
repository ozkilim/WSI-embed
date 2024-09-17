import numpy as np
import time
import os
import argparse
import pdb
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
			
			with torch.inference_mode():
				batch = data['img']
				coords = data['coord'].numpy().astype(np.int32)
			

				batch = batch.to(device, non_blocking=True)
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
					# print(f"Features: {features}")

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
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1','prov_giga_path','virchow','virchow_v2','H-optimus-0'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
	
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	  
	model = nn.DataParallel(model) # use all gpus avalable...
	model = model.to(device)

	total = len(bags_dataset)
	  
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		# print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()

		if args.slide_ext == ".tiff":
			wsi = tiffslide.open_slide(slide_file_path) 
		else:
			wsi = openslide.open_slide(slide_file_path) 

		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
			
		try:
			output_file_path = compute_w_loader(output_path, loader=loader, model=model, device=device, model_name=args.model_name, verbose=1)
		except Exception as e:
			print("Error here")
			print(e)
				  
				  
		# print(output_file_path)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



