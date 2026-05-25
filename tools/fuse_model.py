import argparse

import torch


def parse_args():
	parser = argparse.ArgumentParser(description='Fuse image and lidar checkpoints')
	parser.add_argument('--img', required=True, help='Path to the image checkpoint')
	parser.add_argument('--lidar', required=True, help='Path to the lidar checkpoint')
	parser.add_argument('--out', required=True, help='Output checkpoint path')
	return parser.parse_args()


def main():
	args = parse_args()

	img_ckpt = torch.load(args.img, map_location='cpu')
	state_dict1 = img_ckpt['state_dict']

	pts_ckpt = torch.load(args.lidar, map_location='cpu')
	state_dict2 = pts_ckpt['state_dict']
	# pts_head in camera checkpoint will be overwrite by lidar checkpoint
	state_dict1.update(state_dict2)

	merged_state_dict = state_dict1

	save_checkpoint = {'state_dict':merged_state_dict }
	torch.save(save_checkpoint, args.out)


if __name__ == '__main__':
	main()
