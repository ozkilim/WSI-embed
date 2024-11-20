# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
import h5py
from wsi_core.wsi_utils import save_hdf5
import cv2

def sample_coords(coords, max_patches, seed=42):
	"""
	Sample a maximum number of patches from the coords array.

	Parameters:
	coords (np.ndarray): Array of coordinates with shape (n, 2).
	max_patches (int): Maximum number of patches to sample.
	seed (int, optional): Random seed for reproducibility.

	Returns:
	np.ndarray: Array of sampled coordinates with shape (max_patches, 2) if n > max_patches, otherwise the original array.
	"""
	# Set the random seed for reproducibility
	if seed is not None:
		np.random.seed(seed)

	# Get the number of coordinates
	# print(coords.shape)
	num_coords = coords.shape[0]

	# If the number of coordinates exceeds max_patches, sample max_patches coordinates
	if num_coords > max_patches:
		sampled_indices = np.random.choice(num_coords, max_patches, replace=False)

		# print("indicies!",len(sampled_indices))

		sampled_coords = coords[sampled_indices,:]
		
		# print("coords shape",sampled_coords.shape)

		return sampled_coords
	else:
		return coords


def is_marked(image_array):
    # Convert image to BGR format if it's in RGB
    if image_array.shape[-1] == 3:
        img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image_array  # Assuming it's already in BGR

    # Convert to HSV color space
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define blue color range
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([140, 255, 255])

    # Create masks for red and blue colors
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calculate the proportion of red and blue pixels
    red_pixels = cv2.countNonZero(mask_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]

    # Determine if the patch is marked
    if (red_pixels + blue_pixels) / total_pixels > 0.05:  # Adjust threshold as needed
        return True  # Patch has significant red or blue markings
    return False  # Patch is acceptable



def filter_patches(save_path_hdf5):  
    with h5py.File(save_path_hdf5, 'r+') as h5_file:

        print("Looking into file:", save_path_hdf5)
        coords = np.array(h5_file['coords'])
        num_coords = coords.shape[0]
        attrs = dict(h5_file['coords'].attrs)
        # Assuming patches are stored in 'imgs' dataset within the HDF5 file
        imgs = h5_file['imgs']  # Adjust if your dataset name is different
        # Initialize list of indices to keep
        indices_to_keep = []
        for idx in range(num_coords):
            img = imgs[idx]  # Extract the image data

            # Convert image to a format suitable for OpenCV
            img_array = np.array(img)

            # Check if the image has red or blue markings
            if not is_marked(img_array):
                indices_to_keep.append(idx)

        # Now, filter the coords and imgs datasets
        new_coords = coords[indices_to_keep, :]
        new_imgs = imgs[indices_to_keep, :]

        # Delete existing datasets
        del h5_file['coords']
        del h5_file['imgs']
        h5_file.flush()

        # Create new datasets with filtered data
        new_coords_dataset = h5_file.create_dataset('coords', data=new_coords, compression='gzip')
        new_imgs_dataset = h5_file.create_dataset('imgs', data=new_imgs, compression='gzip')

        # Restore attributes
        for key, value in attrs.items():
            new_coords_dataset.attrs[key] = value


def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time   

    # Add assertions to check segmentation was good
    assert WSI_object.contours_tissue is not None, "Segmentation failed: No tissue contours found"
    assert len(WSI_object.contours_tissue) > 0, "Segmentation failed: Empty tissue contours"
    
    # Check if the segmented area is reasonable (e.g., not too small or too large)
    # total_area = sum(cv2.contourArea(contour) for contour in WSI_object.contours_tissue)
    # total_image_area = WSI_object.level_dim[0][0] * WSI_object.level_dim[0][1]
    # assert 0.01 <= total_area / total_image_area <= 0.99, "Segmentation result seems unreasonable: check parameters"

    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)
	# # HERE we re-load and then cut down and re-save...!
	max_patches = kwargs.get('max_patches', None)

	slide_id = kwargs.get('slide_id', None)

	save_path_hdf5 = os.path.join(patch_save_dir, slide_id + '.h5')
	# print(save_path_hdf5) # should exist at this point...
	# filter_patches(save_path_hdf5) 

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

	

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  microns_per_patch_edge=128, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  max_patches=None,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None):
	


	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)
		# # Calculate patch_size dynamically based on microns_per_patch_edge and microns_per_pixel_x
		patch_size = int(microns_per_patch_edge/WSI_object.microns_per_pixel_x)
		# print("Patch size for this slide is", patch_size)
		step_size = patch_size


		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}

			print("Using custom parameters for slide:", slide_id)

			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})
			
			print("Visualization parameters:", current_vis_params)

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})
			
			print("Filter parameters:", current_filter_params)

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})
			
			print("Segmentation parameters:", current_seg_params)

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})
			
			print("Patch parameters:", current_patch_params)

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		print(f"Visualization level set to: {current_vis_params['vis_level']}")
		print(f"Segmentation level set to: {current_seg_params['seg_level']}")

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		print("Keep IDs:", current_seg_params['keep_ids'])
		print("Exclude IDs:", current_seg_params['exclude_ids'])

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		assert df.loc[idx, 'vis_level'] >= 0, "Visualization level should not be negative"
		assert df.loc[idx, 'seg_level'] >= 0, "Segmentation level should not be negative"

		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 
			print(f"Segmentation completed in {seg_time_elapsed:.2f} seconds")

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)
			print(f"Mask saved at: {mask_path}")

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 'max_patches': max_patches,
										 'save_path': patch_save_dir, 'slide_id':slide_id})
			
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object, **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--microns_per_patch_edge', type = int, default=128,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--max_patches',  type = int, default=None,
					help='Maximum number of patches we can take from a WSI. if WSI makes more than this we sample.')

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(args.preset)
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											microns_per_patch_edge=args.microns_per_patch_edge, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, 
											max_patches=args.max_patches,
											patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip)
