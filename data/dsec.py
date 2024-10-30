from glob import glob
import os
import numpy as np
import cv2
from utils.camera import load_calib_data
import h5py
import hdf5plugin
from data.template_dataset import ImageEventDataset
import os.path as osp

class Our_DSEC(ImageEventDataset):
    def get_resolution(self):
        return 480, 640
   
    def get_image(self, side, idx):
        # side: left or right
        # idx: index in a total of image_cnt*hfr_rate images
        if side == "left":
            path = self.left_img_paths[idx]
        elif side == "right":
            path = self.right_img_paths[idx]

        # Rectify        
        img = cv2.imread(path)
        cam = self.cam_li if side == "left" else self.cam_ri
        img = cam.rectify(img, cv2.INTER_LINEAR)
        return img
    
    def get_seqname(self):
        return self.seqname
    
    def get_disparity(self, side, idx):
        # side: left or right
        # idx: index in a total of image_cnt*hfr_rate images
        assert idx % 2 == 0
        if side == "left":
            disp_path = self.disparity_paths[idx // 2]
            image_disparity = cv2.cvtColor(cv2.imread(disp_path), cv2.COLOR_RGB2GRAY).astype(np.int32)
            disparity = self.cam_li.rectify(image_disparity, cv2.INTER_NEAREST) * ((self.cam_li.K_rect[0][0] + self.cam_li.K_rect[1][1]) / (self.cam_li.K[0][0] + self.cam_li.K[1][1]) * self.cam_li.baseline / self.cam_li.baseline_gt)
            return disparity
        else:
            raise NotImplementedError
    
    def load_timestamps(self, timestamp_path):
        # The txt file format is:
        # # exposure_start_timestamp_us, exposure_end_timestamp_us
        # 51805200087, 51805201557
        # 51805250080, 51805251539
        
        start_ts = []
        end_ts = []
        with open(timestamp_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                start, end = line.split(",")
                start_ts.append(int(start) - self.t_offset)
                end_ts.append(int(end) - self.t_offset)
        
        start_ts = start_ts[::]
        end_ts = end_ts[::]

        self.image_timestamps = list(zip(start_ts, end_ts))

    def get_img_timestamps(self, side, idx):
        # Should return (begin_exposure_t, end_exposure_t) of image[side][idx].
        if side == "left":
            return self.image_timestamps[idx]
        else:
            # actually is also left
            return self.image_timestamps[idx]
    
    def rectify_voxel_img(self, side, voxel):
        # side: left or right
        # idx: index in a total of image_cnt images
        # Rectification should be done in this function.
        cam = self.cam_le if side == "left" else self.cam_re
        voxel = cam.rectify(voxel, cv2.INTER_LINEAR)
        return voxel
        
    def get_events(self, side, idx1, idx2):
        # Should return pointers for all events in the side.
        # Returns [ts, xs, ys, ps]
        f = self.left_f if side == "left" else self.right_f
        xs = f["events"]["x"][idx1:idx2]
        ys = f["events"]["y"][idx1:idx2]
        ts = f["events"]["t"][idx1:idx2]
        ps = f["events"]["p"][idx1:idx2]
        return [ts, xs, ys, ps]
    
    def get_event_idx_from_t(self, side, t):
        ms_to_idx = self.ms_to_idx[0] if side == "left" else self.ms_to_idx[1]
        f = self.left_f if side == "left" else self.right_f
        
        t = int(t)
        ms = t // 1000
        lim = ms_to_idx.shape[0]
        search_begin = ms_to_idx[ms - 1] if ms > 0 and ms < lim else 0
        search_end = ms_to_idx[ms + 1] if ms + 1 < lim else -1
        t_all = f["events"]["t"][search_begin:search_end]
        idx = np.searchsorted(t_all, t)
        idx = idx + search_begin
        return int(idx)
    
    def get_input_cnt(self):
        # Should return number of "serious" predictions. In DSEC it is the number of disparity frames. In DavisReal it is the number of images (before HFR).
        return (len(self.disparity_paths) - 1)*2
    
    def __init__(
        self,
        configs
    ):
        self.root = configs.get("root", "/mnt/ssd/dsec/")
        self.seqname = configs.get("seqname", "interlaken_00_c")

        self.disparity_paths = sorted(glob(osp.join(self.root, "train_disparity", self.seqname, "disparity/image", "*.png")))

        self.calib_path = osp.join(self.root, "train_calibration", self.seqname, "calibration/cam_to_cam.yaml")
        self.cam_le, self.cam_li, self.cam_ri, self.cam_re = load_calib_data(self.calib_path)

        left_img_dir = osp.join(self.root, "train_images", self.seqname, "images", "left", "rectified")
        right_img_dir = osp.join(self.root, "train_images", self.seqname, "images", "right", "rectified")
        self.left_img_paths = sorted(glob(osp.join(left_img_dir, "*.png")))
        self.right_img_paths = sorted(glob(osp.join(right_img_dir, "*.png")))

        # All events are in a single, ~800MB file.
        left_ev_path = os.path.join(self.root, "train_events", self.seqname, "events/left/events.h5")
        right_ev_path = os.path.join(self.root, "train_events", self.seqname, "events/right/events.h5")

        # Load basic data
        self.left_f = h5py.File(left_ev_path, 'r')
        self.right_f = h5py.File(right_ev_path, 'r')

        self.ms_to_idx = [
            self.left_f["ms_to_idx"][()],
            self.right_f["ms_to_idx"][()]
        ]
        self.t_offset = self.left_f["t_offset"][()]

        left_timestamp_path = os.path.join(self.root, "train_images", self.seqname, "images/left/exposure_timestamps.txt")
        self.load_timestamps(left_timestamp_path)

        configs["has_gt_disp"] = True
        configs["sample_interval"] = 2
        super(Our_DSEC, self).__init__(
            configs
        )