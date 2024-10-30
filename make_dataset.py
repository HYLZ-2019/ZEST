from data.dsec import Our_DSEC
from data.mvsec import Our_MVSEC
from data.m3ed import Our_M3ED
import os
import argparse
import tqdm
import numpy as np
import cv2

def make_dataset(ds_type, data_root, seqname):
	assert ds_type in ["dsec", "mvsec", "m3ed"], "Dataloaders are only implemented for DSEC, MVSEC, and M3ED datasets."
	if ds_type == "dsec":
		dataset = Our_DSEC(
			{
				"root": data_root,
				"seqname": seqname,
			}
		)
	elif ds_type == "mvsec":
		dataset = Our_MVSEC(
			{
				"root": data_root,
				"seqname": seqname,
				"voxel_img_clip_percentile_min": 2,
				"voxel_img_clip_percentile_max": 98,
				"window_length": 8,
			}
		)
	elif ds_type == "m3ed":
		dataset = Our_M3ED(
			{
				"root": data_root,
				"seqname": seqname,
			}
		)
	return dataset

if __name__ == "__main__":
	# Only convert dataset to left-view temporal differential images and right-view event temporal integral images.
	# Usage: python make_dataset.py --ds_type dsec --data_root data/dsec --output_dir results --split_name interlaken_00_c
	parser = argparse.ArgumentParser()
	parser.add_argument("--ds_type", type=str, default="dsec")
	parser.add_argument("--data_root", type=str, default="data/dsec")
	parser.add_argument("--output_dir", type=str, default="results")
	parser.add_argument("--seq_name", type=str, default="interlaken_00_c")
	args = parser.parse_args()

	dataset = make_dataset(args.ds_type, args.data_root, args.seq_name)

	left_output_dir = os.path.join(args.output_dir, "left_diff")
	right_output_dir = os.path.join(args.output_dir, "right_integ")
	os.makedirs(left_output_dir, exist_ok=True)
	os.makedirs(right_output_dir, exist_ok=True)

	for i in tqdm.tqdm(range(len(dataset))):
		data = dataset[i]
		left_diff_img = data["left_diff"].astype(np.uint8).squeeze()
		right_integ_img = data["right_integ"].astype(np.uint8).squeeze()
		cv2.imwrite(f"{left_output_dir}/{i:06d}.png", left_diff_img)
		cv2.imwrite(f"{right_output_dir}/{i:06d}.png", right_integ_img)
