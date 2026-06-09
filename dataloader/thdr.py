import os
import numpy as np
import random
import json

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data import THDR3K_TOF_CONFIGS, read_depth, read_rgb, read_tof

class THDR3K(Dataset):
    def __init__(self, data_path, data_list, mode, input_size):
        self.mode = mode
        with open(data_list, 'r') as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[self.mode]
        self.data_files = [os.path.join(data_path, sample["filename"]) for sample in self.sample_list]
            
        self.input_width = input_size[0]
        self.input_height = input_size[1]

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)

        data_file = self.data_files[idx]
        image = read_rgb(data_file + "_rgb.jpg")
        depth_raw = read_depth(data_file + "_depth.png")
        tof, cnh = read_tof(data_file + "_tof.json", THDR3K_TOF_CONFIGS)

        sx, sy, size = THDR3K_TOF_CONFIGS["sx"], THDR3K_TOF_CONFIGS["sy"], THDR3K_TOF_CONFIGS["size"]
        zone = np.array([[sx, sy], [sx + size, sy + size]], dtype=np.int32)

        # Only get the TOF-Zone data
        image = image[zone[0][1]:zone[1][1],zone[0][0]:zone[1][0],...]
        depth_raw = depth_raw[zone[0][1]:zone[1][1],zone[0][0]:zone[1][0]]

        if self.mode == "train":
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)
        
        image, depth_raw, tof, cnh, zone = self._to_tensor(image, depth_raw, tof, cnh, zone)
        image = self.normalize(image)
        
        if self.mode == "train":
            sample = {"image": image, "depth": depth_raw, "tof": tof, "cnh": cnh, "zone": zone}
        elif self.mode == "test":
            sample = {"image": image, "tof": tof, "cnh": cnh, "zone": zone, "depth-gt": depth_raw}
        
        return sample
    
    def _to_tensor(self, image, depth, tof, cnh, zone):
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)
        tof = self.to_tensor(tof)
        cnh = self.to_tensor(cnh)
        zone = torch.as_tensor(zone, dtype=torch.int32)

        return image, depth, tof, cnh, zone

    def augment_image(self, image):
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
