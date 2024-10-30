import cv2
import numpy as np
import torch.utils.data as data

from event_voxel_builder import EventVoxelBuilder

# Template parent class.
class ImageEventDataset(data.Dataset):
    def get_resolution(self):
        # Should return H, W
        raise NotImplementedError

    def get_image(self, side, idx):
        # side: left or right
        # idx: index in a total of image_cnt*hfr_rate images
        # Should return 3*H*W
        # not implemented in this class
        raise NotImplementedError
    
    def get_disparity(self, side, idx):
        # side: left or right
        # idx: index in a total of image_cnt*hfr_rate images
        # not implemented in this class
        raise NotImplementedError
    
    def get_img_timestamps(self, side, idx):
        # Should return (begin_exposure_t, end_exposure_t) of image[side][idx].
        raise NotImplementedError
    
    def rectify_voxel_img(self, side, voxel):
        # side: left or right
        # idx: index in a total of image_cnt images
        # not implemented in this class
        # Rectification should be done in this function.
        raise NotImplementedError
    
    def process_voxel(self, side, voxel):
        voxel = self.rectify_voxel_img(side, voxel)
        return voxel
    
    def get_events(self, side, start_idx, end_idx):
        # Should return pointers for all events in the side.
        # Returns [ts, xs, ys, ps]
        raise NotImplementedError
    
    def get_event_idx_from_t(self, side, t):
        # Should return the index of the event closest to t.
        raise NotImplementedError
    
    def get_input_cnt(self):
        # Should return number of "serious" predictions. In DSEC it is the number of disparity frames.
        raise NotImplementedError
    
    def img_idx_to_ev_idx(self, side, idx):
        t = self.get_img_timestamps(side, idx)[0]
        return self.get_event_idx_from_t(side, t)
    
    def add_voxel_bohan(self, x, y, p, t, bins=5):
        ev_delta_t_us_bin = int(np.ceil((t[-1] - t[0] + 1) / bins))
        # Use the rust-speeded-up voxelbuilder by Bohan.
        voxel_builder = EventVoxelBuilder(
            n_time=bins,
            n_row=self.H,
            n_col=self.W,
            timestamp_per_time=ev_delta_t_us_bin,
        )

        timestamps = np.array(t) - t[0]
        
        # Beware of datasets where p is in {0, 1} or {-1, 1}.
        polarity = np.where(p == 1, 1, -1)

        voxel = voxel_builder.build(
            timestamps,
            y.astype(np.int32),
            x.astype(np.int32),
            polarity,
        )

        return voxel
      
    def make_voxel(self, side, img1_t, img2_t):
        # Voxel(t1, t2) is the events between timestamp t1 and t2.

        # The middle exposure is the same in the two voxel types.   
        mid_start_i = self.get_event_idx_from_t(side, img1_t[1])
        mid_end_i = self.get_event_idx_from_t(side, img2_t[0])

        t, x, y, p = self.get_events(side, mid_start_i, mid_end_i)
    
        # Hardcoded: Only one bin.
        ev_delta_t_us_bin = int(t[-1] - t[0])

        #print(f"Using {mid_end_i - mid_start_i} events for mid voxel.", mid_start_i, mid_end_i, f"Delta t: {ev_delta_t_us_bin} us.")

        if ev_delta_t_us_bin <= 0:
            # could happen
            voxel = np.zeros((self.H, self.W), dtype=np.float32)
        else:
            voxel = self.add_voxel_bohan(x, y, p, t, bins=1).sum(axis=0)


        if img1_t[0] < img1_t[1] and img2_t[0] < img2_t[1]:
            # Calculate event integrations in exposure time.
            i1_start_i = self.get_event_idx_from_t(side, img1_t[0])
            i1_end_i = self.get_event_idx_from_t(side, img1_t[1])
            i2_start_i = self.get_event_idx_from_t(side, img2_t[0])
            i2_end_i = self.get_event_idx_from_t(side, img2_t[1])

            discrete_bins = 5
            t, x, y, p = self.get_events(side, i1_start_i, i1_end_i)
            i1_voxel = self.add_voxel_bohan(x, y, p, t, bins=discrete_bins)

            t, x, y, p = self.get_events(side, i2_start_i, i2_end_i)
            i2_voxel = self.add_voxel_bohan(x, y, p, t, bins=discrete_bins)
            
            # v1_sum[i] = i1_voxel[:i, :].sum(axis=0)
            v1_sum = np.cumsum(i1_voxel, axis=0)

            # v2_sum[i] = i2_voxel[:i, :].sum(axis=0) + [all events from i1_begin_i to i2_begin_i]
            i2_voxel[0] += i1_voxel.sum(axis=0)
            i2_voxel[0] += voxel
            v2_sum = np.cumsum(i2_voxel, axis=0)

            part_1 = np.log(np.mean(np.exp(v1_sum), axis=0))
            part_2 = np.log(np.mean(np.exp(v2_sum), axis=0))
            voxel_new = part_2 - part_1
            
            # Note: the original voxel variable is replaced by part_2 - part_1.            
            voxel = voxel_new

        # Do normalization and rectification
        voxel, _, _ = self.make_norm_nobias(voxel, clip_percentile_min=self.voxel_img_clip_percentile_min, clip_percentile_max=self.voxel_img_clip_percentile_max)
        voxel = self.process_voxel(side, voxel)
        return voxel


    def get_seqname(self):
        raise NotImplementedError

    def __init__(
        self,
        configs
    ):
        super(ImageEventDataset, self).__init__()

        self.H, self.W = self.get_resolution()
        self.has_gt_disp = configs.get("has_gt_disp", False)
        self.MAX_DISP = configs.get("max_disp", 512)

        self.voxel_bin_cnt = configs.get("voxel_bin_cnt", 5)
        self.voxel_img_clip_percentile_min = configs.get("voxel_img_clip_percentile_min", 1)
        self.voxel_img_clip_percentile_max = configs.get("voxel_img_clip_percentile_max", 99)

        # If self.window_length is N, then for frame i, we use [frame i+N - frame i] and [events from frame i to frame i+N] as input.
        self.window_length = configs.get("window_length", 1)

        self.frame_cnt = self.get_input_cnt()

        self.sample_begin = configs.get("sample_begin", 0)
        self.sample_end = configs.get("sample_end", self.frame_cnt-1)
        assert self.sample_end <= self.frame_cnt
        assert self.sample_begin >= 0
        assert self.sample_end > self.sample_begin

        # DSEC needs to have self.sample_interval == 2, because the disparity is only available every 2 frames.
        self.sample_interval = configs.get("sample_interval", 1)

        self.sample_list = []

        for idx in range(self.sample_begin, self.sample_end):                
            # ref_idx is the index of the disparity.
            i1_idx = idx
            i2_idx = min(self.sample_end, idx + self.window_length)
            sample = (i1_idx, i2_idx)
            self.sample_list.append(sample)
        self.sample_list = self.sample_list[::self.sample_interval]

        print("Number of samples: ", len(self.sample_list))
    
    # Normalize a distribution to [0, 255], in which negative values are mapped to [0, 127.5] and positive values are mapped to [127.5, 255].
    def make_norm_nobias(self, img, clip_percentile_min=1, clip_percentile_max=99, clip_val_max = None):
        pos_mask = img > 0

        # Use percentile normalization to prevent extreme values
        # Note: Do not write code like:
        '''
        pos_mask = np.where(img > 0, 1, 0)
        max_pos = np.percentile(img[pos_mask], clip_percentile_max)
        '''
        # Because the above code is *really* slow!

        max_pos = np.percentile(img, clip_percentile_max)
        max_neg = np.percentile(img, clip_percentile_min)
        max_pos = max(max_pos, 1e-3)
        max_neg = min(max_neg, -1e-3) # 负数
        
        if clip_val_max is not None:
            # If there are no extreme values, don't clip
            max_pos = max(max_pos, clip_val_max)
            max_neg = min(max_neg, -clip_val_max)

        img = np.clip(img, max_neg, max_pos)
        max_pos = np.max(img)
        max_neg = np.min(img)
        if max_pos > 0.001:
            img = np.where(pos_mask, img / max_pos, img)
        if max_neg < -0.001:
            img = np.where(1 - pos_mask, img / np.abs(max_neg), img)
        
        img = (img + 1) * 127.5
        return img, max_pos, max_neg

    def clip_log(self, arr):
        if len(arr.shape) == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = np.log1p(arr.astype(np.float32)) # add 1 to prevent log(0)
        return arr
     
    # Temporal differential image
    def get_log_diff_img(self, img1, img2):
        img1 = self.clip_log(img1)
        img2 = self.clip_log(img2)
        diff = img2 - img1
        # Images usually don't have extreme noise points.
        diff, _max_pos, _max_neg = self.make_norm_nobias(diff, clip_percentile_min=1, clip_percentile_max=99)  
        diff = diff[np.newaxis, :, :]
        return diff

    def img_to_gray(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
        
    def __getitem__(self, index):
        output = {}
        
        sample = self.sample_list[index]
        begin_idx, end_idx = sample

        left_img1 = self.get_image("left", begin_idx)
        left_img2 = self.get_image("left", end_idx)
        diff_img_left = self.get_log_diff_img(left_img1, left_img2)
        output["left_diff"] = diff_img_left

        begin_time = self.get_img_timestamps("left", begin_idx)
        end_time = self.get_img_timestamps("left", end_idx)
        right_voxel_img = self.make_voxel("right", begin_time, end_time)
        output["right_integ"] = right_voxel_img

        output["left_image_1"] = left_img1

        if self.has_gt_disp:
            left_disp = self.get_disparity("left", begin_idx)
            left_disp_valid = (left_disp < self.MAX_DISP) * (1 - (left_disp == 0))

            output["gt_disp"] = left_disp
            output["gt_valid"] = left_disp_valid
        
        output["seqname"] = self.get_seqname()
        output["index"] = begin_idx

        return output

    def __len__(self):
        return len(self.sample_list)