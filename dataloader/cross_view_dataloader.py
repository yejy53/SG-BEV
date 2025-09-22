import os
import h5py
import json
import torch
import numpy as np
import torch.nn.functional as F
import random
from PIL import Image, ImageFilter
from torch.utils import data
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from skimage.io import imread

# 归一化转换
normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class DatasetLoader(data.Dataset):
    def __init__(self, cfg, split='train'):
        self.split = split
        self.root = os.path.join(cfg['root'], split)

        # 数据集文件夹路径
        self.gt_folder = cfg['gt_folder']
        self.sate_folder = cfg['sate_folder']
        self.svi_folder = cfg['svi_folder']
        
        # 加载文件名
        self.files_sate = sorted(os.listdir(os.path.join(self.root, self.sate_folder)))
        self.files_svi = sorted(os.listdir(os.path.join(self.root, self.svi_folder)))
        self.files_gt = sorted(os.listdir(os.path.join(self.root, self.gt_folder)))
        
        # 确保数据对齐
        _files_sate_jpg = [f.replace('.png', '.jpg') for f in self.files_sate]
        assert _files_sate_jpg == self.files_svi, "卫星图像与街景图像文件名不匹配"
        assert len(self.files_sate) == len(self.files_gt) > 0, "数据集大小不匹配"
        
        self.envs = np.array([os.path.splitext(f)[0] for f in self.files_sate])
        self.available_idx = np.arange(len(self.files_sate))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]
        file_sate = self.files_sate[env_index]
        file_svi = self.files_svi[env_index]
        
        # 读取图像数据
        img_sate = imread(os.path.join(self.root, self.sate_folder, file_sate))
        img_svi = imread(os.path.join(self.root, self.svi_folder, file_svi))
        semmap_gt = imread(os.path.join(self.root, self.gt_folder, file_sate))
        # gt 应该为0 -- num_classes-1
        semmap_gt = semmap_gt.astype(np.int64)
        semmap_gt = imread(os.path.join(self.root, self.gt_folder, file_sate))
        
        
        # 归一化图像
        img_sate_norm = self.normalize_image(img_sate)
        img_svi_norm = self.normalize_image(img_svi)
        
        return img_sate_norm, img_svi_norm, semmap_gt, file_sate
    
    @staticmethod
    def normalize_image(image):
        """ 归一化图像数据并转换为张量 """
        image = image.astype(np.float32) / 255.0
        image = torch.FloatTensor(image).permute(2, 0, 1)
        image = normalize(image)
        return image