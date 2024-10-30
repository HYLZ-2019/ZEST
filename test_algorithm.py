from make_dataset import make_dataset
from merge_results import merge_disp, map_disp_to_color, map_gt_disp_to_color
import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import tqdm
import numpy as np

from crestereo.nets.crestereo import CREStereo
from transformers import pipeline


def load_cres():
	cres = CREStereo(max_disp=256, mixed_precision=False, test_mode=True)
	cres_pretrained_path = 'crestereo/models/crestereo_eth3d.pth'
	cres_state_dict = torch.load(cres_pretrained_path, weights_only=True)
	cres.load_state_dict(cres_state_dict)
	cres.eval()
	cres = cres.cuda()
	# Freeze the parameters of the CREStereo model
	for param in cres.parameters():
		param.requires_grad = False
	return cres

# Does operation on single pair of numpy images.
# Batch size > 1 can be supported but is not.
def cres_inference(cres, imgL, imgR, n_iter=20):
	# Adjust to (1, 3, H, W)
	imgL = torch.Tensor(imgL).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
	imgR = torch.Tensor(imgR).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).cuda()
	if imgL.shape[1] == 1:
		imgL = imgL.repeat(1, 3, 1, 1)
	if imgR.shape[1] == 1:
		imgR = imgR.repeat(1, 3, 1, 1)
		
	# Pad imgL and imgR so height and weight % 8 == 0
	H, W = imgL.shape[2], imgL.shape[3]
	MOD = 8
	pad_h = (MOD - (imgL.shape[2] % MOD)) % MOD
	pad_w = (MOD - imgL.shape[3] % MOD) % MOD
	half_pad_h = pad_h // 2
	half_pad_w = pad_w // 2
	imgL = F.pad(imgL, (half_pad_w, pad_w-half_pad_w, half_pad_h, pad_h-half_pad_h), mode="constant", value=127.5)
	imgR = F.pad(imgR, (half_pad_w, pad_w-half_pad_w, half_pad_h, pad_h-half_pad_h), mode="constant", value=127.5)
	# imgL and imgR should be of shape (B, 3, H, W)
	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	with torch.inference_mode():
		pred_flow_dw2 = cres(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
		pred_flow = cres(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	
	pred_disp = pred_flow[0, 0, half_pad_h:H+half_pad_h, half_pad_w:W+half_pad_w]
	pred_disp = pred_disp.cpu().numpy()
	return pred_disp

def DA_inference(pipe, img_arr):
	image = Image.fromarray(img_arr)
	depth = pipe(image)["depth"]
	depth = np.array(depth)
	return depth

def load_DAv2():
	pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda")
	return pipe

if __name__ == "__main__":
	# Only convert dataset to left-view temporal differential images and right-view event temporal integral images.
	# Usage: python make_dataset.py --ds_type dsec --data_root data/dsec --output_dir results --split_name interlaken_00_c
	parser = argparse.ArgumentParser()
	parser.add_argument("--ds_type", type=str, default="dsec")
	parser.add_argument("--data_root", type=str, default="data/dsec")
	parser.add_argument("--output_dir", type=str, default="results")
	parser.add_argument("--seq_name", type=str, default="interlaken_00_c")
	parser.add_argument("--visualize", action="store_true")
	args = parser.parse_args()

	dataset = make_dataset(args.ds_type, args.data_root, args.seq_name)
	print(f"Inferencing on {args.ds_type} dataset {args.seq_name} with {len(dataset)} samples...")

	final_disp_output_dir = os.path.join(args.output_dir, "final_disp")
	visualize_output_dir = os.path.join(args.output_dir, "visualize")
	os.makedirs(final_disp_output_dir, exist_ok=True)
	os.makedirs(visualize_output_dir, exist_ok=True)

	cres_model = load_cres()
	da_pipe = load_DAv2()

	MAE = []

	for i in tqdm.tqdm(range(len(dataset))):
		data = dataset[i]
		left_diff_img = data["left_diff"].astype(np.uint8).squeeze()
		right_integ_img = data["right_integ"].astype(np.uint8).squeeze()
		index = data["index"]
		gt = data["gt_disp"]
		gt_valid = data["gt_valid"]
		left_img = data["left_image_1"]
		
		stereo = cres_inference(cres_model, left_diff_img, right_integ_img)
		mono = DA_inference(da_pipe, left_diff_img)

		final = merge_disp(stereo, mono, left_diff_img)
		
		gt_mask = np.where(gt_valid)
		MAE.append(np.mean(np.abs(final[gt_mask] - gt[gt_mask])))

		np.save(f"{final_disp_output_dir}/{index:06d}.npy", final)

		if args.visualize:
			# All in one row: left_image_1, left_diff, right_integ, stereo, mono, final, gt
			
			gt_viz, min_disp, max_disp = map_gt_disp_to_color(data["gt_disp"])
			
			# Visualize them with the same colormap
			stereo_viz = map_disp_to_color(stereo, min_disp, max_disp)
			mono_viz = map_disp_to_color(mono, min_disp, max_disp)
			final_viz = map_disp_to_color(final, min_disp, max_disp)
			
			left_diff_viz = cv2.cvtColor(left_diff_img, cv2.COLOR_GRAY2BGR)
			right_integ_viz = cv2.cvtColor(right_integ_img, cv2.COLOR_GRAY2BGR)

			# Add text on top of each image
			for img, text in zip([left_img, left_diff_viz, right_integ_viz, stereo_viz, mono_viz, final_viz, gt_viz], ["Left Image", "Left Diff", "Right Integ", "Stereo", "Mono", "Final", "GT"]):
				cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			
			# Concatenate them
			concat = np.concatenate([left_img, left_diff_viz, right_integ_viz, stereo_viz, mono_viz, final_viz, gt_viz], axis=1)
			
			cv2.imwrite(f"{visualize_output_dir}/{index:06d}.png", concat)

	print(f"MAE: {np.mean(MAE)}")

			
