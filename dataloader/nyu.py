import os
import numpy as np
import random
import json

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data import read_depth, read_rgb, simulate_tof


class NYUv2(Dataset):
    def __init__(self, data_path, data_list, mode, input_size,
                 zone_size=52, zone_grid_rows=8, zone_grid_cols=8,
                 sim_cnh_bins=18, sim_cnh_range=5.4, sim_dis_max=4.0):
        self.mode = mode
        with open(data_list, 'r') as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[self.mode]
        self.data_files = [os.path.join(data_path, sample["filename"]) for sample in self.sample_list]

        self.input_width = input_size[0]
        self.input_height = input_size[1]

        # Zone configuration
        self.zone_size = zone_size
        self.zone_rows = zone_grid_rows
        self.zone_cols = zone_grid_cols

        # Simulation parameters
        self.sim_cnh_bins = sim_cnh_bins
        self.sim_cnh_range = sim_cnh_range
        self.sim_dis_max = sim_dis_max

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)

        data_file = self.data_files[idx]
        
        image = read_rgb(data_file + "_rgb.jpg")
        depth_raw = read_depth(data_file + "_depth.png")

        # Simulate the TOF related data.
        if self.mode == "train":
            # Randomly apply data augmentation & flip
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)
            do_flip = random.random()
            if do_flip > 0.5:
                image, depth_raw = self.flip_image(image, depth_raw)
            do_rotate = random.random()
            if do_rotate > 0.5:
                image, depth_raw = self.rotate_image(image, depth_raw, max_angle=5)

            zone = self._get_random_tof_zone(depth_raw.shape)
        else:
            zone = self._get_certain_tof_zone(depth_raw.shape)
        
        tof, cnh = simulate_tof(
            depth=depth_raw,
            zone=zone,
            rows=self.zone_rows,
            cols=self.zone_cols,
            bins=self.sim_cnh_bins,
            cnh_range_max=self.sim_cnh_range,
            dis_max=self.sim_dis_max,
        )

        # TODO: Keep for the future work. It's used when the whole image and the ToF-Zone both are needed for training. But currently we only use the ToF-Zone data, so the random crop is not applied.
        # Randomly crop the data to fit the input size.
        # image, depth_raw, zone = self.random_crop(image, depth_raw, zone, self.input_width, self.input_height)

        # Only get the TOF-Zone data
        image = image[zone[0][1]:zone[1][1],zone[0][0]:zone[1][0],...]
        depth_raw = depth_raw[zone[0][1]:zone[1][1],zone[0][0]:zone[1][0]]            

        image, depth_raw, tof, cnh, zone = self._to_tensor(image, depth_raw, tof, cnh, zone)
        image = self.normalize(image)
        
        if self.mode == "train":
            sample = {"image": image, "depth": depth_raw, "tof": tof, "cnh": cnh, "zone": zone}
        elif self.mode == "test":
            sample = {"image": image, "tof": tof, "cnh": cnh, "zone": zone, "depth-gt": depth_raw}
        
        return sample
    
    def _get_random_tof_zone(self, shape):
        h, w = shape
        zone_height = self.zone_size * self.zone_rows
        zone_width = self.zone_size * self.zone_cols
        if zone_height > h or zone_width > w:
            raise ValueError(
                f"ToF zone ({zone_width}x{zone_height}) exceeds depth shape ({w}x{h})"
            )
        y0 = random.randint(0, h - zone_height)
        x0 = random.randint(0, w - zone_width)
        return np.array([[x0, y0], [x0 + zone_width, y0 + zone_height]], dtype=np.int32)

    def _get_certain_tof_zone(self, shape):
        h, w = shape
        zone_height = self.zone_size * self.zone_rows
        zone_width = self.zone_size * self.zone_cols
        if zone_height > h or zone_width > w:
            raise ValueError(
                f"ToF zone ({zone_width}x{zone_height}) exceeds depth shape ({w}x{h})"
            )
        y0 = (h - zone_height) // 2
        x0 = (w - zone_width) // 2
        return np.array([[x0, y0], [x0 + zone_width, y0 + zone_height]], dtype=np.int32)
    
    def _to_tensor(self, image, depth, tof, cnh, zone):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        tof = self.to_tensor(tof)
        cnh = self.to_tensor(cnh)
        zone = torch.as_tensor(zone, dtype=torch.int32)

        return image, depth, tof, cnh, zone

    @staticmethod
    def augment_image(image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    @staticmethod
    def flip_image(image, depth):
        image = np.flip(image, axis=1).copy()
        depth = np.flip(depth, axis=1).copy()
        return image, depth

    @staticmethod
    def rotate_image(image, depth, max_angle=5, fill_depth=0.0):
        angle = random.uniform(-max_angle, max_angle)
        h, w = depth.shape

        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        image_rot = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        depth_rot = cv2.warpAffine(
            depth,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_depth,
        )

        return image_rot.astype(np.float32), depth_rot.astype(np.float32)

    @staticmethod
    def random_crop(image, depth_raw, zone, w, h):
        img_h, img_w = image.shape[:2]

        # Calculate the valid region of crop. ToF-Zone should always be included after crop.
        dx = min(zone[0][0], img_w - w, img_w - zone[1][0])
        dy = min(zone[0][1], img_h - h, img_h - zone[1][1])
        x_offset = max(0, zone[1][0] - w)
        y_offset = max(0, zone[1][1] - h)

        # The left-top point of the crop region.
        x = random.randint(0, dx) + x_offset
        y = random.randint(0, dy) + y_offset

        if x >= zone[0][0]:
            x = zone[0][0]
        if y >= zone[0][1]:
            y = zone[0][1]
            
        assert x <= zone[0][0] and y <= zone[0][1]

        # The data after crop.
        image_crop = image[y:y + h, x:x + w]
        depth_raw_crop = depth_raw[y:y + h, x:x + w]
        zone_crop = [[zone[0][0] - x, zone[0][1] - y], 
                     [zone[1][0] - x, zone[1][1] - y]]
        
        return np.array(image_crop, dtype=np.float32), np.array(depth_raw_crop, dtype=np.float32), np.array(zone_crop, dtype=np.int32)
