import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='preset_builder')
parser.add_argument('--preset_name', type=str,
					help='name of preset')
parser.add_argument('--seg_level', type=int, default=-1, 
					help='downsample level at which to segment')
parser.add_argument('--sthresh', type=int, default=8, 
					help='segmentation threshold')
parser.add_argument('--mthresh', type=int, default=7, 
					help='median filter threshold')
parser.add_argument('--use_otsu', action='store_true', default=False)
parser.add_argument('--close', type=int, default=4, 
					help='additional morphological closing')
parser.add_argument('--a_t', type=int, default=100, 
					help='area filter for tissue')
parser.add_argument('--a_h', type=int, default=16, 
					help='area filter for holes')
parser.add_argument('--max_n_holes', type=int, default=8, 
					help='maximum number of holes to consider for each tissue contour')
parser.add_argument('--vis_level', type=int, default=-1, 
					help='downsample level at which to visualize')
parser.add_argument('--line_thickness', type=int, default=250, 
					help='line_thickness to visualize segmentation')
parser.add_argument('--white_thresh', type=int, default=5, 
					help='saturation threshold for whether to consider a patch as blank for exclusion')
parser.add_argument('--black_thresh', type=int, default=50, 
					help='mean rgb threshold for whether to consider a patch as black for exclusion')
parser.add_argument('--no_padding', action='store_false', default=True)
parser.add_argument('--contour_fn', type=str, choices=['four_pt', 'center', 'basic', 'four_pt_hard'], default='four_pt',
					help='contour checking function')
parser.add_argument('--remove_markings', action='store_true', default=False,
					help='Enable removal of colored markings from WSI during segmentation')
parser.add_argument('--disable_red_removal', action='store_true', default=False,
					help='Disable red color removal even when remove_markings is enabled')
parser.add_argument('--disable_blue_removal', action='store_true', default=False,
					help='Disable blue color removal even when remove_markings is enabled')
parser.add_argument('--disable_green_removal', action='store_true', default=False,
					help='Disable green color removal even when remove_markings is enabled')
parser.add_argument('--disable_black_removal', action='store_true', default=False,
					help='Disable black/near-black color removal even when remove_markings is enabled')
parser.add_argument('--disable_off_white_removal', action='store_true', default=False,
					help='Disable off-white color removal even when remove_markings is enabled')


if __name__ == '__main__':
	args = parser.parse_args()
	seg_params = {'seg_level': args.seg_level, 'sthresh': args.sthresh, 'mthresh': args.mthresh, 
				  'close': args.close, 'use_otsu': args.use_otsu, 'keep_ids': 'none', 'exclude_ids': 'none',
				  'remove_markings': args.remove_markings}
	
	# Add marking colors configuration if marking removal is enabled
	if args.remove_markings:
		marking_colors = {
			'red': {
				'enabled': not args.disable_red_removal,  # Enabled by default, disabled only if requested
				'lower1': [0, 100, 80],
				'upper1': [10, 255, 255],
				'lower2': [170, 100, 80],
				'upper2': [180, 255, 255]
			},
			'blue': {
				'enabled': not args.disable_blue_removal,  # Enabled by default
				'lower': [100, 100, 80],
				'upper': [130, 255, 255]
			},
			'green': {
				'enabled': not args.disable_green_removal,  # Enabled by default
				'lower': [35, 5, 5],
				'upper': [90, 255, 255]
			},
			'off_white': {
				'enabled': not args.disable_off_white_removal,  # Enabled by default
				'lower': [0, 0, 200],
				'upper': [179, 30, 255]
			},
			'black': {
				'enabled': not args.disable_black_removal,  # Enabled by default
				'lower': [0, 0, 0],
				'upper': [179, 255, 80]
			}
		}
		
		# Validate that at least one color is enabled
		enabled_colors = [color for color, config in marking_colors.items() if config['enabled']]
		if not enabled_colors:
			raise ValueError("Error: remove_markings is enabled but no colors are enabled for removal. "
							"At least one color must be enabled when using marking removal.")
		
		print(f"Preset: Marking removal enabled for colors: {', '.join(enabled_colors)}")
		
		# Convert marking colors to a flat dictionary for CSV storage
		seg_params.update({
			'red_enabled': marking_colors['red']['enabled'],
			'red_lower1': str(marking_colors['red']['lower1']),
			'red_upper1': str(marking_colors['red']['upper1']),
			'red_lower2': str(marking_colors['red']['lower2']),
			'red_upper2': str(marking_colors['red']['upper2']),
			'blue_enabled': marking_colors['blue']['enabled'],
			'blue_lower': str(marking_colors['blue']['lower']),
			'blue_upper': str(marking_colors['blue']['upper']),
			'green_enabled': marking_colors['green']['enabled'],
			'green_lower': str(marking_colors['green']['lower']),
			'green_upper': str(marking_colors['green']['upper']),
			'off_white_enabled': marking_colors['off_white']['enabled'],
			'off_white_lower': str(marking_colors['off_white']['lower']),
			'off_white_upper': str(marking_colors['off_white']['upper']),
			'black_enabled': marking_colors['black']['enabled'],
			'black_lower': str(marking_colors['black']['lower']),
			'black_upper': str(marking_colors['black']['upper'])
		})
	
	filter_params = {'a_t':args.a_t, 'a_h': args.a_h, 'max_n_holes': args.max_n_holes}
	vis_params = {'vis_level': args.vis_level, 'line_thickness': args.line_thickness}
	patch_params = {'white_thresh': args.white_thresh, 'black_thresh': args.black_thresh, 
					'use_padding': args.no_padding, 'contour_fn': args.contour_fn}

	all_params = {}
	all_params.update(seg_params)
	all_params.update(filter_params)
	all_params.update(vis_params)
	all_params.update(patch_params)
	params_df = pd.DataFrame(all_params, index=[0])
	params_df.to_csv('presets/{}'.format(args.preset_name), index=False)
	