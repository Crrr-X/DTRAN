from __future__ import annotations
import torch
from torch.utils.data import Dataset
from src.data.utils import transform
from osgeo import gdal

gdal.DontUseExceptions()

class DemDataset(Dataset):

    def __init__(self, data_path: str, mode: str = "train", crop_size: int = 96, scale: int = 4):

        super(DemDataset, self).__init__()
        self.mode = mode
        with open(data_path, "r", errors='ignore') as lines: 
            self.samples = []
            for line in lines:
                hr_path = line.strip().split(" ")[0]
                lr_path = line.strip().split(" ")[1]
                feats = line.strip().split(" ")[2:]  
                self.samples.append([hr_path, lr_path, feats])
        self.samples.sort()
        if mode == "train":
            self.transform_scale = transform.Compose(transform.RandomScaleCrop(crop_size=crop_size, scale=scale), 
                                                     transform.RandomHorizontalFlip(),
                                                     transform.RandomVerticalFlip(),
                                                     transform.RandomRotation(),
                                                     transform.ToTensor(mode=mode))
        if mode == "val":
            self.transform = transform.Compose(transform.ToTensor(mode=mode))
            self.reverse = False
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        sample = self.samples[index]
        if len(sample) == 2:
            hr_path, lr_path = sample
            feats_path = []  
        else:
            hr_path, lr_path, feats_path = sample

        if not hr_path or not lr_path:
            raise ValueError
        hr = gdal.Open(hr_path).ReadAsArray()
        lr = gdal.Open(lr_path).ReadAsArray()
        feats = []
        if feats_path and len(feats_path): 
            for feat in feats_path:
                feats.append(gdal.Open(feat).ReadAsArray())
        if self.mode == "train":
            hr, lr, feats = self.transform_scale(hr, lr, feats)
        if self.mode == "val":
            hr, lr, feats = self.transform(hr, lr, feats)

        return hr, lr, feats

    def __len__(self) -> int:
        return len(self.samples)
