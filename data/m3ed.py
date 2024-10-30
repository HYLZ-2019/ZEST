import torch
import glob
import os
import numpy as np
import cv2
import h5py
import hdf5plugin
from data.template_dataset import ImageEventDataset
import os.path as osp


def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P

def read_intrinsic(f):
    camera_model = f["calib/camera_model"][()]
    distortion_coeffs = f["calib/distortion_coeffs"][()]
    distortion_model = f["calib/distortion_model"][()]
    intrinsics = f["calib/intrinsics"][()]
    resolution = f["calib/resolution"][()]
    K = np.eye(3)
    K[0,0] = intrinsics[0]
    K[1,1] = intrinsics[1]
    K[0,2] = intrinsics[2]
    K[1,2] = intrinsics[3]
    T_to_prophesee_left = f["calib/T_to_prophesee_left"][()]
    return K, distortion_coeffs, T_to_prophesee_left

def get_map(source_group, target_group, move_x=True):
    
    target_T_to_prophesee_left = target_group['T_to_prophesee_left'][...]
    source_T_to_prophesee_left = source_group['T_to_prophesee_left'][...]

    source_T_target = source_T_to_prophesee_left @ np.linalg.inv( target_T_to_prophesee_left )
    target_T_source = np.linalg.inv(source_T_target)

    target_model = target_group['camera_model']
    target_dist_coeffs = target_group['distortion_coeffs'][...]
    target_dist_model = target_group['distortion_model']
    target_intrinsics = target_group['intrinsics'][...]
    target_res = target_group['resolution'][...]
    target_Size = target_res

    target_K = np.eye(3)
    target_K[0,0] = target_intrinsics[0]
    target_K[1,1] = target_intrinsics[1]
    target_K[0,2] = target_intrinsics[2]
    target_K[1,2] = target_intrinsics[3]

    target_P = np.zeros((3,4))
    target_P[:3,:3] = target_K

    source_model = source_group['camera_model']
    source_dist_coeffs = source_group['distortion_coeffs'][...]
    source_dist_model = source_group['distortion_model']
    source_intrinsics = source_group['intrinsics'][...]
    source_res = source_group['resolution'][...]
    source_width, source_height = source_res
    source_Size = source_res

    source_K = np.eye(3)
    source_K[0,0] = source_intrinsics[0]
    source_K[1,1] = source_intrinsics[1]
    source_K[0,2] = source_intrinsics[2]
    source_K[1,2] = source_intrinsics[3]

    source_P = np.zeros((3,4))
    source_P[:3,:3] = target_K
    source_P[0,3] =  target_K[0,0] * target_T_source[0,3]
    source_P[1,3] =  target_K[1,1] * target_T_source[1,3]
    if not move_x:
        source_P[0,3] = 0

    #target_R = target_T_source[:3,:3] # np.eye(3)
    #source_R = np.eye(3) # target_T_source[:3,:3]
    target_R = np.eye(3)
    source_R = source_T_target[:3,:3]

    map_target = np.stack(cv2.initUndistortRectifyMap(target_K, target_dist_coeffs, target_R, target_P, target_Size, cv2.CV_32FC1), axis=-1)
    map_source = np.stack(cv2.initUndistortRectifyMap(source_K, source_dist_coeffs, source_R, source_P, target_Size, cv2.CV_32FC1), axis=-1)
    inv_map_target = invert_map(map_target)
    inv_map_source = invert_map(map_source)
    return map_source, map_target, inv_map_source, inv_map_target

class Our_M3ED(ImageEventDataset):

    def __init__(
        self,
        configs
    ):
        self.root = configs.get("root", "/mnt/ssd/m3ed/")
        self.seqname = configs.get("seqname", "car_urban_day_horse")
        data_f_path = configs.get("data_path", os.path.join(self.root, f"{self.seqname}_data.h5"))
        gt_f_path = configs.get("gt_path", data_f_path.replace("_data.h5", "_depth_gt.h5"))
        self.data_f = h5py.File(data_f_path, 'r')
        self.gt_f = h5py.File(gt_f_path, 'r')

        self.load_timestamps()
        self.calibrate()
        
        configs["has_gt_disp"] = True
        super(Our_M3ED, self).__init__(configs)

    # Warning: the rectification of the M3ED dataloader is problematic.
    # In the camera system of M3ED, the event cameras and image cameras are not at the same height:
    # Img      Img
    # Evs      Evs
    # So the baseline between left-image and right-event is NOT parallel to the ground. To rectify the data *right*, the rectified images should all look tilted.
    # However, depth prediction models do not deal with tilted images well. They usually assume that the ground (and everything else) is horizontal. As a result, when we rectify the data *wrong*, causing the rectified images to look horizontal, the models can actually work better.
    # As a result, we choose the wrong rectification method. Please try to find a better way (or use a better dataset) if you'd like to follow our work.
    def calibrate(self):
        f = self.data_f
        map_re, _, inv_map_re, _ = get_map(f["/prophesee/right/calib"], f["/prophesee/left/calib"], move_x=False)
        map_li, _, inv_map_li, _ = get_map(f["/ovc/left/calib"], f["/prophesee/left/calib"])
        
        self.map_left = map_li
        self.map_right = map_re

        # Calculate disparity between prophesee left & prophesee right from depth.
        # Disparity = f * B / Z
        focal_length = f["/prophesee/left/calib/intrinsics"][0]
        # Calculate the baseline from the transformation matrix from prophesee left to prophesee right.
        baseline = np.linalg.norm(f["/prophesee/right/calib/T_to_prophesee_left"][0:3, 3])
        self.depth_to_disp = focal_length * baseline
    
    def rectify_voxel_img(self, side, voxel):
        assert side == "right"
        voxel = cv2.remap(voxel, self.map_right[..., 0], self.map_right[..., 1], cv2.INTER_LINEAR)
        return voxel
    
    def get_seqname(self):
        return self.seqname
    
    def get_resolution(self):
        return 720, 1280
    
    def load_timestamps(self):
        self.disp_ts = self.gt_f["ts"][()]
        self.img_ts = self.data_f["/ovc/ts"][()]
        self.min_time = min(self.disp_ts[0], self.img_ts[0])
        self.max_time = max(self.disp_ts[-1], self.img_ts[-1])

        img_idx_to_disp_idx = np.searchsorted(self.disp_ts, self.img_ts)
        img_idx_to_disp_idx = np.clip(img_idx_to_disp_idx, 0, len(self.disp_ts) - 1)
        self.img_idx_to_disp_idx = img_idx_to_disp_idx

    def get_img_timestamps(self, side, idx):
        # No exposure information is given, let it be 0
        return self.img_ts[idx], self.img_ts[idx]
    
    def get_disparity(self, side, idx):
        assert side == "left"
        disp_idx = self.img_idx_to_disp_idx[idx]
        gt_depth = self.gt_f["/depth/prophesee/left"][disp_idx]
        gt_disp = self.depth_to_disp / gt_depth
        return gt_disp
    
    def get_input_cnt(self):
        return self.data_f["/ovc/left/data"].shape[0]

    def get_image(self, side, i):
        assert side == "left", "Not implemented"
        img = self.data_f["/ovc/left/data"][i]
        img = cv2.remap(img, self.map_left[..., 0], self.map_left[..., 1], cv2.INTER_LINEAR)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    
    def remap_disp(self, disp):
        # Remap the non-zero points. Don't do linear interpolation.
        # get the coordinates of non-zero points
        ii, jj = np.nonzero(disp)
        projected_i = self.inv_map_disp[..., 1][ii, jj].astype(int)
        projected_j = self.inv_map_disp[..., 0][ii, jj].astype(int)
        vals = disp[ii, jj]

        filter = (projected_i >= 0) & (projected_i < 720) & (projected_j >= 0) & (projected_j < 1280)
        projected_i = projected_i[filter]
        projected_j = projected_j[filter]
        vals = vals[filter]

        new_disp = np.zeros_like(disp)
        # assign the values to the new image
        new_disp[projected_i, projected_j] = vals
        return new_disp
    
    def get_event_idx_from_t(self, side, t):
        assert side == "right"
        prev_img_idx = np.searchsorted(self.img_ts, t, side='left')
        prev_img_idx = min(prev_img_idx, len(self.img_ts) - 1)
        next_img_idx = prev_img_idx + 1

        prev_ev_idx = self.data_f["/ovc/ts_map_prophesee_right_t"][prev_img_idx]
        if next_img_idx < len(self.img_ts):
            next_ev_idx = self.data_f["/ovc/ts_map_prophesee_right_t"][next_img_idx]
        else:
            next_ev_idx = -1
        
        ev_idx = np.searchsorted(self.data_f["/prophesee/right/t"][prev_ev_idx:next_ev_idx], t, side='left')

        return int(ev_idx + prev_ev_idx)

    def get_events(self, side, idx1, idx2):
        assert side == "right"
        xs = self.data_f["/prophesee/right/x"][idx1:idx2]
        ys = self.data_f["/prophesee/right/y"][idx1:idx2]
        ts = self.data_f["/prophesee/right/t"][idx1:idx2]
        ps = self.data_f["/prophesee/right/p"][idx1:idx2]
        return [ts, xs, ys, ps]