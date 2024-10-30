import numpy as np
import cv2
import glob

def visualize_depth(depth):
    # Make inf 0
    mask = np.isfinite(depth)
    masked_max = np.max(depth[mask])
    masked_min = np.min(depth[mask])
    d = np.clip(depth, masked_min, masked_max)
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    d = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)
    d = np.where(mask[..., None], d, np.zeros_like(d))
    return d


def visualize(imgdir):
	for path in sorted(glob.glob(imgdir + "*.npy")):
		disp = np.load(path)
		viz = visualize_depth(disp)
		out_path = path.replace(".npy", ".png")
		cv2.imwrite(out_path, viz)
