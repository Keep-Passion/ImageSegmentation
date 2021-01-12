import os
import h5py
from skimage import io
import numpy as np
from libtiff import TIFF
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import warnings
from utils import *
from model.losses.bcw_loss import *
from model.losses.uw_loss import *


# The first step of deep learning experiment is data preprocessing.   

def data_preprocess(dataset_name: str, in_path: Path, out_path: Path, zfill_num: int = 3, suffix: str = 'png') -> None:
    if dataset_name == 'isbi2012':
        __prepare_data_isbi2012(in_path, out_path, zfill_num, suffix)
    
    
    
def __prepare_data_isbi2012(in_path: Path, out_path: Path, zfill_num: int = 3, imgs_suffix: str = 'png') -> None:
    """
        Prepare the data for training
        The data can be download from ISBI2012 challange (website: 'http://brainiac2.mit.edu/isbi_challenge/')
    """

    imgs_inpath = Path(in_path, 'train-volume.tif')
    labels_inpath = Path(in_path, 'train-labels.tif')
    imgs_tif = TIFF.open(imgs_inpath, mode='r')
    labels_tif = TIFF.open(labels_inpath, mode='r')

    imgs_outpath = Path(out_path, 'images')
    labels_outpath = Path(out_path, 'labels')
    imgs_outpath.mkdir(parents=True, exist_ok=True) 
    labels_outpath.mkdir(parents=True, exist_ok=True)
    print("File Converting...")
    print("The image's path is {}".format(str(imgs_outpath  )))
    print("The label's path is {}".format(str(labels_outpath)))
        
    files_name = []
    idx = -1
    for img, label in zip( list(imgs_tif.iter_images()),  list(labels_tif.iter_images()) ):
        idx += 1
        files_name.append(str(idx).zfill(zfill_num) + '.' + imgs_suffix)
        io.imsave(str(Path(imgs_outpath, str(idx).zfill(zfill_num) + '.' + imgs_suffix)), img)
        np.save(str(Path(labels_outpath, str(idx).zfill(zfill_num) + '.npy')), (label/255)) # cell is 1 and border is 0
        if idx == 0:
            plt.figure(figsize=(10, 10))
            plt.subplot(1,2,1), plt.imshow(img, cmap='gray'),   plt.title('image'), plt.axis("off")
            plt.subplot(1,2,2), plt.imshow(label, cmap='gray'), plt.title('label'), plt.axis("off")
            plt.show()            
    print("File Convert Done!")


def prepare_weight_map(labels_path: Path, out_path: Path, func_names: List, class_num :int = 2) -> None:
    """
        Generate the weight map in data directory, such as balanced class weight or unet weight
    """
    file_paths = []
    for file in labels_path.rglob("*.npy"):
        if Path.is_file(file) and '.ipynb_checkpoints' not in str(str(file)):
            file_paths.append(file)
    file_paths.sort()
    # init
    bcw_weight = BalancedClassWeight(class_num)
    uw_weight  = UnetWeight()
    print("Weight Map Caculating...")
    for file_path in file_paths:
        file_name = file_path.name
        print("Analysing {} ..." .format(file_name))
        label = load_img(file_path)
        for func_name in func_names:
            weight_path = Path(out_path, func_name)
            weight_path.mkdir(parents=True, exist_ok=True)
            if func_name == 'bcw':
                weight = bcw_weight._get_weight(label)
            elif func_name == 'uw':
                assert class_num == 2, "The unet weight only support binary segmentation"
                weight = uw_weight._get_weight(label)
            print("{} for {} has done".format(func_name, file_name))
            print("Visualization:")
            for vis_idx in range(class_num):
                print("{} {}, the max value is {} and the min value is {}".format(func_name, vis_idx, 
                                                                                  np.amax(weight[:, :, vis_idx]),
                                                                                  np.amin(weight[:, :, vis_idx])))
                    # print('    and the unique values are {}'.format(np.unique(weight[:, :, vis_idx])))
            plt.figure(figsize=(10, 10))
            plt.subplot(1,class_num + 1,1), plt.imshow(label, cmap="gray"), plt.title("label"), plt.axis("off")
            for vis_idx in range(class_num):
                plt.subplot(1,class_num + 1,vis_idx + 2) 
                plt.imshow(weight[:, :, vis_idx]), plt.title("{} {}".format(func_name, vis_idx)), plt.axis("off")
            plt.show()
            # print(str(Path(weight_path, file_name.replace('png', 'npy'))))
            np.save(str(Path(weight_path, file_name.replace("png", "npy"))), weight)
