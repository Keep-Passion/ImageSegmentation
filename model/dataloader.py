import os, time, random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
from skimage import morphology
import torchvision.transforms as tr
from utils import *
from typing import List, Tuple, Dict
import torchvision.transforms as tr


class SegDataset(Dataset):
    """
        Segmentation Dataset, is suit to 2d and 3d images
    """
#     def __init__(self, data_dir: Path, files_names: List, weight_dir=None, crop_size=None, 
#                  use_augment=[False, False, False], depth=None, norm_transform=None, 
#                  is_load_data_to_memory: bool = True) -> None:
    def __init__(self, params: Dict, files_names: List, use_augment: bool = True) -> None:        
        
        self.params = params
        
        self._img_paths = []
        self._label_paths = []
        self._weight_paths = []
        for item in files_names:
            self._img_paths.append(   Path(Path(self.params["data"]["dir"]), 'images', item))
            self._label_paths.append( Path(Path(self.params["data"]["dir"]), 'labels', item.split('.')[0] + '.npy'))
            self._weight_paths.append(Path(Path(self.params["data"]["dir"]), self.params["data"]["weights_name"], item.split('.')[0] + '.npy'))
        
        self._use_augment = use_augment
        self._depth = params["data"]["imgs_depth"]  # if the images is 3D structure,  this param recode the depth of the 3D images
        self._augmentor = DataAugmentor(params["train"]["crop_size"], self._depth, self._use_augment)
                              
        self._norm_transform = tr.Compose([
                                tr.ToTensor(),
                                tr.Normalize(mean = [self.params["data"]["imgs_mean"]], 
                                             std =  [self.params["data"]["imgs_std"]])
                            ])
        
        img = load_data_from_disk(self._img_paths[0])
        label = load_data_from_disk(self._label_paths[0])
        self._h, self._w = img.shape[:2]
        self._class_num = self.params["data"]["labels_class_num"]
        
        self._imgs_mem, self._labels_mem, self._weights_mem = None, None, None
        self.is_load_data_to_memory = self.params["data"]["is_load_data_to_cpu_mem"]
        if self.is_load_data_to_memory:
            self._imgs_mem    = load_data_to_memory(self._img_paths)
            self._labels_mem  = load_data_to_memory(self._label_paths)
            self._weights_mem = load_data_to_memory(self._weight_paths)
    
    def __getitem__(self, index: int) -> Dict:
        # read images
        if self._depth is None:    # 2D analysis 
            img    = load_data(self._imgs_mem,    self._img_paths,    index)  
            label  = load_data(self._labels_mem,  self._label_paths,  index)
            weight = load_data(self._weights_mem, self._weight_paths, index)
        else:                      # 3D analysis
            img    = np.zeros((self._h, self._w, self._depth))
            label  = np.zeros((self._h, self._w, self._depth))
            weight = np.zeros((self._h, self._w, self._depth, self._class_num))
            
            for idx, depth_idx in enumerate(range(index, index + self._depth)):
                if depth_idx > len(self._img_path):   # it will copy the last image to form 3D image
                    img[: , :, idx]       = load_data(self._imgs_mem,    self._img_paths,    len(self._img_path) - 1)
                    label[: , :, idx]     = load_data(self._labels_mem,  self._label_paths,  len(self._img_path) - 1)
                    weight[: , :, idx, :] = load_data(self._weights_mem, self._weight_paths, len(self._img_path) - 1)
                else:
                    img[: , :, idx]       = load_data(self._imgs_mem,    self._img_paths,    depth_idx)
                    label[: , :, idx]     = load_data(self._labels_mem,  self._label_paths,  depth_idx)
                    weight[: , :, idx, :] = load_data(self._weights_mem, self._weight_paths, depth_idx)
        
        # data augmentation
        if len(self._weight_paths) != 0:
            in_transfer = [img, label, weight]
            data = self._augmentor.start_augmentation(in_transfer, self.params["train"]["train_augment"])
            img, label, weight = data[0], data[1], data[2]

        else: 
            in_transfer = [img, label]
            data = self._augmentor.start_augmentation(in_transfer, self.params["train"]["train_augment"])
            img, label = data[0], data[1]
        
        # to tensor
        if self._depth is None:              # 2D analysis->[C,H,W]
            img = self._norm_transform(img)
            label = torch.from_numpy(label[np.newaxis, :, :])
            if len(self._weight_paths) != 0:
                weight = torch.from_numpy(weight.transpose((2, 0, 1)))
        else:                                # 3D analysis->[C,D,H,W]
            img    = np.ascontiguousarray(img, dtype=np.float32)
            label  = np.ascontiguousarray(label, dtype=np.float32)
            img = self._norm_transform(img)
            img = img.unsqueeze(0)
            label = torch.from_numpy(label.transpose((2, 0, 1))[np.newaxis, :, :, :])
            if len(self._weight_paths) != 0:
                weight = np.ascontiguousarray(weight, dtype=np.float32)
                weight = torch.from_numpy(weight.transpose((3, 2, 0, 1))) # shape(H, W, D, C) -> (C, D, H, W)
        
        out_data = {'img': img, 'label': label}
        if len(self._weight_paths) != 0:
            out_data['weight'] = weight
        return out_data        
    
    def __len__(self) -> int:
        return len(self._img_paths)
    
class DataAugmentor:
    """
        The class of data augmentor, you can add your own function in this class
    """  
    def __init__(self, crop_size: int = None, depth = None, use_augment = True) -> None:
        self.__crop_size   = crop_size
        self.__depth = depth
        self.__use_augment = use_augment

    def start_augmentation(self, data, train_augment: List) -> List:
        if self.__use_augment:
            for item in train_augment:
                if item == 'rand_rotate':
                    data = self.__rand_rotation(data)               # 0, 90, 180, 270
                elif item == 'rand_vflip':
                    data = self.__rand_vertical_flip(data)            # p<0.5, flip
                elif item == 'rand_hflip':
                    data = self.__rand_horizontal_flip(data)          # p<0.5, flip
                elif item == 'rand_zflip' and self.__depth is not None:
                    data = self.__rand_z_filp(data)                   # p<0.5, flip
        data = self.__rand_crop(data)
        return data
 
    def __rand_rotation(self, data) -> List:
        angle = random.choice([0, 90, 180, 270])
        rotate_idx = angle / 90
        data_num = len(data)
        for data_idx in range(data_num):
            data[data_idx] = np.rot90(data[data_idx], rotate_idx).copy()
        return data

    def __rand_vertical_flip(self, data) -> List:
        p = random.random()
        if p < 0.5:
            data_num = len(data)
            for data_idx in range(data_num):
                data[data_idx] = np.flipud(data[data_idx]).copy()
        return data

    def __rand_horizontal_flip(self, data) -> List:
        p = random.random()
        if p < 0.5:
            data_num = len(data)
            for data_idx in range(data_num):
                data[data_idx] = np.fliplr(data[data_idx]).copy()
        return data

    def __rand_z_filp(self, data) -> List:
        p = random.random()
        if p < 0.5:
            data_num = len(data)
            for data_idx in range(data_num):
                data[data_idx] = np.flip(data[data_idx], 2).copy()
        return data

    def __rand_crop(self, data) -> List:
        random_h = random.randint(0, data[0].shape[0] - self.__crop_size)
        random_w = random.randint(0, data[0].shape[1] - self.__crop_size)
        data_num = len(data)
        for data_idx in range(data_num):
            data[data_idx] = data[data_idx][random_h: random_h + self.__crop_size, random_w: random_w + self.__crop_size]    
        return data
