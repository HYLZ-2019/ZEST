import numpy as np
import cv2
import h5py
import hdf5plugin
from data.template_dataset import ImageEventDataset
import os.path as osp
import yaml

class Our_MVSEC(ImageEventDataset):
    def get_resolution(self):
        return 260, 346

    def get_image(self, side, idx):
        # side: left or right
        # idx: index in a total of image_cnt images
        img = self.data_f["davis"][side]["image_raw"][idx]
        # If gray, convert to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if side == "left":
            img = cv2.remap(img, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        elif side == "right":
            img = cv2.remap(img, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        
        return img
    
    def get_seqname(self):
        return self.seqname
    
    def get_disparity(self, side, idx):
        # side: left or right
        # idx: index in a total of image_cnt images
        img_ts = self.image_timestamps["start_ts"][idx]
        closest_idx = np.searchsorted(self.disp_timestamps, img_ts)
        closest_idx = min(len(self.disp_timestamps) - 1, closest_idx)
        depth = self.gt_f["davis"][side]["depth_image_rect"][closest_idx]
        depth = np.where(np.isfinite(depth), depth, 0.0001)
        disparity = self.depth_to_disp_const / depth
        disparity = np.where(disparity < 512, disparity, 0)

        return disparity
    
    def load_timestamps(self):
        base_ts = self.data_f["davis"]["left"]["image_raw_ts"][:]
        self.time_bias = base_ts[0]
        base_ts = base_ts - self.time_bias
        base_ts = base_ts * 1e6

        self.disp_timestamps = (self.gt_f["davis"]["left"]["depth_image_rect_ts"][:] - self.time_bias) * 1e6

        # The left images and right images do not have the same timestamps
        self.right_img_timestamps = (self.data_f["davis"]["right"]["image_raw_ts"][:] - self.time_bias) * 1e6

        # No exposure time is given, let it be 0
        self.image_timestamps = {
            "start_ts": base_ts,
            "end_ts": base_ts,
        }

    def get_img_timestamps(self, side, idx):
        # Should return (begin_exposure_t, end_exposure_t) of image[side][idx].
        if side == "left":
            return (self.image_timestamps["start_ts"][idx], self.image_timestamps["end_ts"][idx])
        else:
            # Also is left
            return (self.image_timestamps["start_ts"][idx], self.image_timestamps["end_ts"][idx])
    
    def rectify_voxel_img(self, side, voxel):
        # side: left or right
        # idx: index in a total of image_cnt images
        # Rectification should be done in this function.
        if side == "left":
            voxel = cv2.remap(voxel, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
        else:
            voxel = cv2.remap(voxel, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)
        return voxel
        
    def get_events(self, side, idx1, idx2):
        # Should return pointers for events in the side.
        # Returns [ts, xs, ys, ps]
        evs = self.data_f["davis"][side]["events"]
        xs = evs[idx1:idx2, 0].astype(np.int16)
        ys = evs[idx1:idx2, 1].astype(np.int16)
        ts = (evs[idx1:idx2, 2] - self.time_bias) * 1e6
        ps = evs[idx1:idx2, 3]
        return [ts, xs, ys, ps]
    
    def get_event_idx_from_t(self, side, t):
        # Should return the index of the event closest to t.
        # Find the index of the image with closest timestamp and use image_raw_event_inds to shorten the search range.
        img_timestamps = self.image_timestamps["start_ts"] if side == "left" else self.right_img_timestamps
        prev_img_idx = np.searchsorted(img_timestamps, t, side="left")
        next_img_idx = prev_img_idx + 1
        if prev_img_idx < len(img_timestamps):
            prev_ev_idx = self.data_f["davis"][side]["image_raw_event_inds"][prev_img_idx]
        else:
            prev_ev_idx = self.data_f["davis"][side]["image_raw_event_inds"][-1]
        if next_img_idx < len(img_timestamps):
            next_ev_idx = self.data_f["davis"][side]["image_raw_event_inds"][next_img_idx]
        else:
            next_ev_idx = -1

        t_all = self.data_f["davis"][side]["events"][prev_ev_idx:next_ev_idx, 2]
        idx = np.searchsorted(t_all, t)
        idx = idx + prev_ev_idx
        return int(idx)
    
    def get_input_cnt(self):
        return self.data_f["davis"]["left"]["image_raw"].shape[0]

    
    def get_rectification_map(self, intrinsics_extrinsics):
        """Produces tables that map rectified coordinates to distorted coordinates.

        x_distorted = rectified_to_distorted_x[y_rectified, x_rectified]
        y_distorted = rectified_to_distorted_y[y_rectified, x_rectified]
        """
        dist_coeffs = intrinsics_extrinsics['distortion_coeffs']
        D = np.array(dist_coeffs)

        intrinsics = intrinsics_extrinsics['intrinsics']
        K = np.array([[intrinsics[0], 0., intrinsics[2]],
                    [0., intrinsics[1], intrinsics[3]], [0., 0., 1.]])
        K_new = np.array(intrinsics_extrinsics['projection_matrix'])[0:3, 0:3]

        R = np.array(intrinsics_extrinsics['rectification_matrix'])

        size = (intrinsics_extrinsics['resolution'][0],
                intrinsics_extrinsics['resolution'][1])

        rectified_to_distorted_x, rectified_to_distorted_y = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, K_new, size, cv2.CV_32FC1)

        return rectified_to_distorted_x, rectified_to_distorted_y


    def __init__(
        self,
        configs
    ):
        self.root = configs.get("root", "x")
        self.seqname = configs.get("seqname", "x")

        self.split_name = self.seqname[:-1]
        self.split_num = self.seqname[-1:]

        data_f_path = osp.join(self.root, self.split_name, f"{self.seqname}_data.hdf5")
        self.data_f = h5py.File(data_f_path, 'r')
        gt_f_path = osp.join(self.root, self.split_name, f"{self.seqname}_gt.hdf5")
        self.gt_f = h5py.File(gt_f_path, 'r')

        self.calib_path = osp.join(self.root, self.split_name, f"{self.split_name}_calib/camchain-imucam-{self.split_name}.yaml")
        self.calib_info = yaml.load(open(self.calib_path, "r"), Loader=yaml.FullLoader)
        baseline = -self.calib_info['cam1']['T_cn_cnm1'][0][3]
        focal_length = self.calib_info['cam0']['intrinsics'][0]
        self.depth_to_disp_const = baseline * focal_length
        self.load_timestamps()

        self.left_map_x, self.left_map_y = self.get_rectification_map(self.calib_info['cam0'])
        self.right_map_x, self.right_map_y = self.get_rectification_map(self.calib_info['cam1'])

        configs["has_gt_disp"] = True
        super(Our_MVSEC, self).__init__(
            configs
        )