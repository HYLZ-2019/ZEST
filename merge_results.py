import numpy as np
import cv2
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import torch.nn.functional as F
import os
import argparse

device = "cuda"

def showarr(arr, name):
	plt.imshow(arr)
	plt.colorbar()
	plt.savefig(f"tmp/{name}.png")
	plt.close()

class MergeDataset(torch.utils.data.Dataset):
	def __init__(self, mono_disp_root, stereo_disp_root, left_grad_root, re_size=None):
		self.mono_disps = sorted(glob.glob(f"{mono_disp_root}/*.npy"))
		self.stereo_disps = sorted(glob.glob(f"{stereo_disp_root}/*.npy"))
		self.left_grads = sorted(glob.glob(f"{left_grad_root}/*.png"))
		self.re_size = re_size
		
	def __len__(self):
		return min(len(self.mono_disps), len(self.stereo_disps), len(self.left_grads))
	
	def __getitem__(self, idx):
		mono_disp = np.load(self.mono_disps[idx])
		stereo_disp = np.load(self.stereo_disps[idx])
		left_grad = cv2.imread(self.left_grads[idx], cv2.IMREAD_GRAYSCALE)
		if self.re_size is not None:
			new_H, new_W = self.re_size
			mono_disp = cv2.resize(mono_disp, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
			stereo_disp = cv2.resize(stereo_disp, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
			left_grad = cv2.resize(left_grad, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
		
		return {
			"stereo_pred": torch.tensor(stereo_disp).float().unsqueeze(0),
			"mono_pred": torch.tensor(mono_disp).float().unsqueeze(0),
			"left_grad": torch.tensor(left_grad).float().unsqueeze(0),
			"index": idx
		}
	

class TwoScaleNet(torch.nn.Module):
	def __init__(self, configs, stereo_pred, mono_pred, left_grad):
		super(TwoScaleNet, self).__init__()

		self.configs = configs

		self.mono_pred = mono_pred

		self.grad_mask = torch.abs(left_grad - 127.5)
		self.init_pool_size = configs["init_pool_size"]

		self.avgpool = torch.nn.AvgPool2d(self.init_pool_size, stride=1, padding=self.init_pool_size//2)
		self.init_scale = self.mean_pool_init(stereo_pred, mono_pred+1, self.grad_mask)

		# pad
		scale = torch.zeros_like(mono_pred)
		self.merge_scale = torch.nn.Parameter(scale.clone().detach(), requires_grad=True)

		bias = torch.ones_like(mono_pred)
		self.merge_bias = torch.nn.Parameter(bias.clone().detach(), requires_grad=True)

	def mean_pool_init(self, stereo, mono, grad_mask):
		pool_target = self.avgpool(stereo * grad_mask)
		pool_mono = self.avgpool(mono * grad_mask)
		res = pool_target / pool_mono
		return res

	def forward(self):
		merged = (self.mono_pred + self.merge_bias) * (self.merge_scale + self.init_scale)
		params = [self.merge_scale + self.init_scale, self.merge_bias]
		return merged, params
		
class TwoScaleLoss(torch.nn.Module):
	def __init__(self, configs):
		super(TwoScaleLoss, self).__init__()
		total_weight = configs["stereo_weight"] + configs["scale_grad_weight"] + configs["scale_reg_weight"]
		self.stereo_weight = configs["stereo_weight"] / total_weight
		self.scale_grad_weight = configs["scale_grad_weight"] / total_weight
		self.scale_reg_weight = configs["scale_reg_weight"] / total_weight
		self.min_scale = configs["min_scale"]
		self.max_scale = configs["max_scale"]
		self.edge_aware_k = configs["edge_aware_k"]
		self.sigmoid = torch.nn.Sigmoid()

		# A x-gradient filter and a y-gradient filter
		self.x_filter = torch.tensor([[0, 0, 0], [1, -1, 0], [0, 0, 0]]).to(device).float().unsqueeze(0).unsqueeze(0)
		self.y_filter = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).to(device).float().unsqueeze(0).unsqueeze(0)

	def calc_grad(self, x):
		grad_x = torch.nn.functional.conv2d(x, self.x_filter, padding=1)
		grad_y = torch.nn.functional.conv2d(x, self.y_filter, padding=1)
		return torch.sqrt(grad_x**2 + grad_y**2) / 255
	
	def L2(self, x, y):
		return torch.sqrt((x - y)**2)
	
	def scale_reg(self, scale):
		min = self.min_scale
		max = self.max_scale
		dist_1 = torch.clamp(min-scale, 0, 100)
		dist_2 = torch.clamp(scale-max, 0, 100)
		return dist_1**2 + dist_2**2

	def forward(self, merged_disp, params, stereo_pred, mono_pred, left_grads):

		scale, bias = params

		scale_grad_loss = 0
		for kernel in [self.x_filter, self.y_filter]:
			for scale, bias, midas in [(scale, bias, mono_pred)]:
				scale_grad = (torch.abs(F.conv2d(scale, kernel, padding=1)) / 255) 
				bias_grad = (torch.abs(F.conv2d(bias, kernel, padding=1)) / 255)
				mono_grad =(torch.abs(F.conv2d(midas, kernel, padding=1)) / 255)

				# Scale & bias may change where the depth changes, whether in stereo_pred or mono_pred
				grad_weight = torch.exp(-(
					(mono_grad)*self.edge_aware_k)**2) + 0.01
		
				scale_grad_loss += (scale_grad + bias_grad) * grad_weight

		# Where events are dense, encourage merged_disp to follow stereo_pred using L2 Loss
		event_weights = torch.clamp(torch.abs(left_grads - 127.5) - 10, 0, 255) / 255
		stereo_loss = torch.abs(stereo_pred - merged_disp) * event_weights

		# Give penalty to scale out of range [self.min_scale, self.max_scale], especially negative scales
		scale_range_loss = self.scale_reg(scale)

		loss = \
			self.stereo_weight * stereo_loss.mean() + \
			self.scale_grad_weight * scale_grad_loss.mean() + \
			self.scale_reg_weight * scale_range_loss.mean()

		return loss, stereo_loss, scale_grad_loss, scale_range_loss

def map_disp_to_color(disp, minval=0, maxval=255):
	assert disp.ndim == 2
	disp = np.clip(disp, minval, maxval)
	
	disp = (disp - minval) / (maxval - minval)
	disp = 1 - disp
	cmap = mpl.cm.get_cmap("jet")
	colored = cmap(disp)[:, :, :3]
	# Convert it to uint8
	colored = (colored * 255).astype(np.uint8)
	return colored

# For sparse disparity maps, leave the 0 values as 0, and map the rest to color
def map_gt_disp_to_color(disp):
	assert disp.ndim == 2
	mask = disp > 0
	masked_max = np.max(disp[mask])
	masked_min = np.min(disp[mask])
	if masked_max - masked_min < 1e-3:
		masked_max = masked_min + 1e-3

	mask_disp = np.clip(disp, masked_min, masked_max)
	mask_disp = (mask_disp - masked_min) / (masked_max - masked_min)
	mask_disp = 1 - mask_disp

	cmap = mpl.cm.get_cmap("jet")
	colored = cmap(mask_disp)[:, :, :3]
	# Convert it to uint8
	colored = (colored * 255).astype(np.uint8)
	colored = np.where(mask[..., None], colored, np.zeros_like(colored))
	
	return colored, masked_min, masked_max

def map_disp_error_to_color(disp_ori, gt, compare=None, minval=0, maxval=100):
	assert disp_ori.ndim == 2
	disp = np.clip(disp_ori, minval, maxval)
	
	disp = (disp - minval) / (maxval - minval)
	cmap = mpl.cm.get_cmap("jet")
	colored = cmap(disp)[:, :, :3]
	# Convert it to uint8
	colored = (colored * 255).astype(np.uint8)
	
	mask = gt > 0
	# For each dot where gt is not 0, calculate np.abs(disp-gt), and write it to the colored image
	error = np.abs(disp_ori - gt)
	compare_error = np.abs(compare - gt)

	# Enumerate through all mask==1 dots
	# make zip(*np.where(mask)) a list
	ls = [(i, j) for i, j in zip(*np.where(mask))]
	for i, j in ls[::5]:
		if compare_error[i, j] < error[i, j] - 5:
			cv2.putText(colored, "+", (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
		elif compare_error[i, j] > error[i, j] + 5:
			cv2.putText(colored, "-", (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
	return colored

def iterative_optimize_twoimg(configs, batch):
	b, c, h, w = batch["stereo_pred"].shape
	assert b == 1, "The code does not support batch size > 1. Optimization results will be wrong."
	PAD = 8
	pad_to_h = (h + PAD - 1) // PAD * PAD
	pad_to_w = (w + PAD - 1) // PAD * PAD
	pad_h = pad_to_h - h
	pad_w = pad_to_w - w

	stereo_t = torch.nn.functional.pad(batch["stereo_pred"], (0, pad_w, 0, pad_h), mode="constant", value=batch["stereo_pred"].mean().item()).squeeze(0)
	mono_t = torch.nn.functional.pad(batch["mono_pred"], (0, pad_w, 0, pad_h), mode="constant", value=batch["mono_pred"].mean().item()).squeeze(0)
	
	left_grad_t = torch.nn.functional.pad(batch["left_grad"], (0, pad_w, 0, pad_h), mode="constant", value=batch["left_grad"].mean().item()).squeeze(0)

	stereo_t, mono_t, left_grad_t = stereo_t.to(device), mono_t.to(device), left_grad_t.to(device)
	indexes = batch["index"]

	# Do the merging
	merged = TwoScaleNet(configs, stereo_t, mono_t, left_grad_t).to(device)
	m_loss = TwoScaleLoss(configs).to(device)
	
	optimizer_type = configs["optimizer"]
	if optimizer_type == "adam":
		optimizer = torch.optim.Adam(merged.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-4)
	elif optimizer_type == "sgd":
		optimizer = torch.optim.SGD(merged.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
	elif optimizer_type == "rms":
		optimizer = torch.optim.RMSprop(merged.parameters(), lr=1e-3, alpha=0.9, eps=1e-6, weight_decay=1e-4)
	else:
		print("Optimizer not supported, using Adam instead.")
		optimizer = torch.optim.Adam(merged.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=1e-4)
	
	all_lr = configs["all_lr"]
	steps_1 = len(all_lr)

	if steps_1 == 0:
		# Do nothing to the stereo predictions
		return batch["stereo_pred"]
	
	for i in range(steps_1):
		optimizer.param_groups[0]['lr'] = all_lr[i]
		optimizer.zero_grad()
		merged_disp_1, params = merged()
		loss, stereo_loss, mono_loss, scale_reg_loss = m_loss(merged_disp_1, params, stereo_t, mono_t, left_grad_t)
		loss.backward()
		# Normalize gradients.
		torch.nn.utils.clip_grad_norm_(merged.parameters(), 1)
		optimizer.step()
	
	# TODO: add MAX_DISPARITY as a parameter instead of hardcoding 255
	merged_disp = merged_disp_1[0].detach().cpu().numpy().squeeze().clip(0, 255)
	# Do unpadding
	merged_disp = merged_disp[:h, :w]

	return merged_disp

# This function is used in the pipeline of test_algorithm.py.
def merge_disp(stereo_pred, mono_pred, left_grad):
	batch = {
		"stereo_pred": torch.Tensor(stereo_pred).unsqueeze(0).unsqueeze(0).cuda(),
		"mono_pred": torch.Tensor(mono_pred).unsqueeze(0).unsqueeze(0).cuda(),
		"left_grad": torch.Tensor(left_grad).unsqueeze(0).unsqueeze(0).cuda(),
		"index": 0
	}
	ITERS = 100
	configs = {
		"init_pool_size": 5,
		"stereo_weight": 1,
		"scale_grad_weight": 1000000,
		"scale_reg_weight": 5,
		"min_scale": 0,
		"max_scale": 3,
		"edge_aware_k": 500,
		"optimizer": "adam", # "adam", "sgd", "rms"
		"all_lr": [1e-0]*ITERS + [1e-1]*ITERS + [1e-2]*ITERS + [1e-3]*ITERS,
		"viz": False
	}
	merged = iterative_optimize_twoimg(configs, batch)
	return merged

if __name__ == "__main__":
	# Use with the following command:
	# python merge_results.py --mono_disp_dir data/mono_results --stereo_disp_dir data/stereo_results --left_grad_dir data/left_diff --output_dir data/merged_results
	parser = argparse.ArgumentParser(description="Merge disparity results from mono and stereo predictions.")
	parser.add_argument("--mono_disp_dir", type=str, required=True, help="Directory containing mono disparity results.")
	parser.add_argument("--stereo_disp_dir", type=str, required=True, help="Directory containing stereo disparity results.")
	parser.add_argument("--left_grad_dir", type=str, required=True, help="Directory containing left gradient images.")
	parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged results.")
	args = parser.parse_args()

	dataset = MergeDataset(
		args.mono_disp_dir,
		args.stereo_disp_dir,
		args.left_grad_dir
	)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=24)
	
	ITERS = 100
	configs = {
		"init_pool_size": 5,
		"stereo_weight": 1,
		"scale_grad_weight": 1000000,
		"scale_reg_weight": 5,
		"min_scale": 0,
		"max_scale": 3,
		"edge_aware_k": 500,
		"optimizer": "adam", # "adam", "sgd", "rms"
		"all_lr": [1e-0]*ITERS + [1e-1]*ITERS + [1e-2]*ITERS + [1e-3]*ITERS,
		"viz": False
	}

	os.makedirs(args.output_dir, exist_ok=True)
	
	for batch in tqdm.tqdm(dataloader):
		merged = iterative_optimize_twoimg(configs, batch)
		index = batch["index"].item()[0]
		np.save(os.path.join(args.output_dir, f"{index:06d}.npy"), merged)