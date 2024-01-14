import os
from pathlib import Path
import h5py
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline

import cv2, time
import random
import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import accelerator, Accelerator
from ema_pytorch import EMA
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from sklearn.model_selection import KFold
from skimage import morphology
# import pandas as pd
from torch.optim import Adam

from utils import *

from model.segtrainer import SegTrainer
from model.seginferencer import SegInferencer
from model.losses import losser

from ddpm.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


# you can write your own func of data_preprocess begin at training
def data_preprocess(in_dir: Path, out_dir: Path, suffix: str = "png", labels_class_num: int = 2) -> Tuple:
    """
        Prepare the data for training
        The data can be download from ISBI2012 challange (website: 'http://brainiac2.mit.edu/isbi_challenge/')
        :param in_dir:  the input directory of origin data
        :param out_dir: the output directory of processed data
        :param suffix:  the suffix of raw image file
        :param labels_class_num: the number of class in label
        :return: (mean, std, num) of raw image, which is used for normalization
    """

    # can not read libtif in numpy 0.19 and libtiff 0.4
    imgs_inpath = Path(in_dir, "images")
    labels_inpath = Path(in_dir, "labels")

    imgs_outpath = Path(out_dir, "images")
    labels_outpath = Path(out_dir, "labels")
    imgs_outpath.mkdir(parents=True, exist_ok=True)
    labels_outpath.mkdir(parents=True, exist_ok=True)

    print("File Converting...")
    print("The image's path is {}".format(str(imgs_outpath)))
    print("The label's path is {}".format(str(labels_outpath)))

    imgs_path_list = list(imgs_inpath.glob("*.png"))

    for img_idx, img_path in enumerate(imgs_path_list):
        img = io.imread(str(img_path))
        img   = (img  - img.min()) * 255.0 / (img.max() - img.min()) # normalization to [0,1]
        label = Image.open(str(Path(labels_inpath, img_path.name)))
        im = label.convert('1')  # 图像二值化
        label_npy = np.array(im)

        io.imsave(str(Path(imgs_outpath, str(img_path.name).replace(".png", '.' + suffix))), img.astype(np.uint8))
        np.save(str(Path(labels_outpath, str(img_path.name).replace(".png", '.npy'))), label_npy)

        label = np.load(str(Path(labels_outpath, str(img_path.name).replace(".png", '.npy'))))

        if img_idx == 0:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1), plt.imshow(img, cmap="gray"), plt.title("image"), plt.axis("off")
            plt.subplot(1, 2, 2), plt.imshow(label, cmap="gray"), plt.title("label"), plt.axis("off")
            plt.show()

    # it is useful for your data preprocessing function
    print("Generating weight map...")
    # weights_func = ["bcw", "uw"]
    weights_func = ["bcw"]
    losser.prepare_weight_map(labels_outpath, out_dir, weights_func, labels_class_num)
    print("Generated weight map in {}".format(str(out_dir)))

    mean, std, num = count_mean_and_std(imgs_outpath, suffix)
    print("mean = {}, std ={} and image num = {}".format(mean, std, num))
    print("Data Preprocess has done")

    return mean, std, num


# you can define your own data_postprocess in the step of inference
def data_postprocess(in_img: np.ndarray) -> np.ndarray:
    """
        post processing data after inference before evaluation
        :param in_img:
        :return out_img:
    """
    # skeletonization of result
    # out_img = morphology.skeletonize(1 - in_img, method="lee") /255 # Lee Skeleton method
    # dialiation with 3 pixels
    # out_img = 1 - morphology.dilation(out_img, morphology.square(3))
    return in_img


def train(ddpm_model, data_name: str, data_bk_dir: Path, data_dir: Path, params_path: Path, custom_desc: str, pre_process: bool, accelerator) -> str:
    """
        Train netowrk
        :param data_name:
        :param data_bk_dir:  the input directory of origin data
        :param data_dir:     the output directory of processed data
        :param params_path:  the path of certain experimental parameters
        :param custom_desc:  the custom description which add last in the experiment_name, you can define it
        :return: experiment_name, which is used to record all the parameter in training
    """

    # load params of experiment
    params = ParamsYml().load_params(params_path)

    # data preprocess
    if pre_process:
        mean, std, num = data_preprocess(data_bk_dir, data_dir, suffix=params["data"]["imgs_suffix"],
                                         labels_class_num=params["data"]["labels_class_num"])


    # ddpm training
    ddpm = ddpm_model
    ddpm_opt = Adam(ddpm.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # ddpm_path = r'/guocancheng/denoising-diffusion-pytorch/results_1222/model-best.pt' # predict v
    # ddpm_path = r'/guocancheng/denoising-diffusion-pytorch/results_0105/model-best.pt'  # predict x0
    # data = torch.load(ddpm_path, map_location=accelerator.device)
    # ddpm_opt.load_state_dict(data['opt'])
    # if accelerator.is_main_process:
    #     ddpm_ema = EMA(ddpm, beta=0.995, update_every=4)
    #     ddpm_ema.to(accelerator.device)
    #     ddpm_ema.load_state_dict(data["ema"])
    # else:
    #     ddpm_ema = None
    # ddpm.load_state_dict(data['model'])
    # del data


    # you can direct assign values in here in order to avoid repeat execution of data_preprocess
    # for isibi2012, mean = 0.49475772084753505, std = 0.1678160259016179 and image num = 30
    params["data"]["name"] = data_name
    params["data"]["dir"] = str(data_dir)
    params["data"]["imgs_mean"] = 0.4864331404857399
    params["data"]["imgs_std"] = 0.2203333372738792
    params["data"]["imgs_num"] = 220

    experiment_name = set_experiment_name(params, custom_desc)

    experiment_dir = Path(Path.cwd(), "experiment", experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print("Parameters will be saved in {}".format(str(experiment_dir)))

    seg_trainer = SegTrainer(params, experiment_dir, data_postprocess, ddpm, ddpm_opt, accelerator)

    # Training
    seg_trainer.start_training()

    return experiment_name

def build_ddpm():
    unet = Unet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    )

    ddpm = GaussianDiffusion(
        unet,
        image_size=256,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective='pred_x0'
    )
    return ddpm

def inference(ddpm_model, experiment_name: str, accelerator) -> None:
    """
        inference data according to previous training record
        :param experiment_name: which is used to record all the parameter in training
    """
    # Wether to analyse all data or only test dataset
    is_test_all = True
    seg_trainer = SegInferencer(ddpm_model, experiment_name, accelerator, is_test_all, data_postprocess)
    seg_trainer.start_inference()


if __name__ == '__main__':
    data_name = "alla"
    data_bk_dir = Path("../dualData", data_name, "raw/train")
    data_dir = Path(Path.cwd(), "data", data_name, "train")
    params_path = Path(Path.cwd(), "alla_bcw.yml")
    custom_desc = "ddpm_seg_0112_1822"  # you can add your description in here

    # Tranining
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='no'
    )

    ddpm_model = build_ddpm()
    experiment_name = train(ddpm_model, data_name, data_bk_dir, data_dir, params_path, custom_desc, False, accelerator)

    # Inference
    # experiment_name = "seg_alla_unet_loss_bcw_ddpm_seg_0112_1822"
    inference(ddpm_model, experiment_name, accelerator)