import os
import random
import numpy as np
import torch
import math
from pathlib import Path
from skimage import io
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from typing import Callable, Iterable, List, Set, Tuple, Dict
from IPython.display import clear_output
from skimage import morphology
import matplotlib.pyplot as plt

import model.metrics as metrics
from .nets.networks import Network
from .visualization import *

from utils import *

                                
class SegInferencer:
    """
        Inference class
    """
    def __init__(self, experiment_name: str, is_test_all: bool = True, data_postprocess = None) -> None:
        
        self.experiment_name = experiment_name
        self.experiment_dir = Path(Path.cwd(), "experiment", self.experiment_name)
        params_path = Path(self.experiment_dir, "experiment_params.yml")
        self.params = ParamsYml().load_params(params_path)
        self.is_test_all = is_test_all
        self.data_postprocess = data_postprocess
        
        # model parameters
        self.net_class = Network(self.params["train"]["model_name"])
        self.model = None
        
        # gpu parameters
        self.gpu_detector = GpuDetector()
        try:
            self.gpu_detector.set_gpu_device(self.params["gpu"]["cuda_devices"])
        except AssertionError as e:
            print(e)
        
        self.logger = Logger(True, Path(self.experiment_dir, "test_log.txt"))
        
    def start_inference(self):
        self.logger.log_print("Starting inference: {}".format(str(self.experiment_dir.name)))
        if self.is_test_all:
            self.logger.log_print("Analyzing all the images")
        else:
            self.logger.log_print("Analyzing only the test images")
        test_metric_list = []
        if self.params["data"]["use_kfold"]:
            for kf_idx in range(self.params["data"]["kf_num"]):
                self.logger.log_print("Start analyzing {}_fold test files".format(str(kf_idx + 1)))
                pth_path = Path(self.experiment_dir, str(self.params["data"]["kf_num"]) + '_fold_' + str(kf_idx), "best_model_state.pth")
                if self.is_test_all:
                    files_list = self.params["data"]["all_files"]
                    test_metric = self.test(pth_path, files_list, 'all_vis')
                else:
                    files_list = self.params["data"]["test_files_list"][kf_idx]
                    test_metric = self.test(pth_path, files_list, 'test_vis')
                self.logger.log_print("{} fold: test_{} is {}".format(str(kf_idx+1), self.params["test"]["metric_name"], test_metric))
                test_metric_list.append(test_metric)  
        else:
            pth_path = Path(self.experiment_dir, "best_model_state.pth")
            if self.is_test_all:
                files_list = self.params["data"]["all_files"]
                test_metric = self.test(pth_path, files_list, 'all_vis')
            else:
                files_list = self.params["data"]["test_files_list"][0]
                test_metric = self.test(pth_path, files_list, 'test_vis')
            test_metric_list.append(test_metric)
        
        if self.params["data"]["use_kfold"]:
            self.logger.log_print("For cross validation, mean is: {}, std is {}".format(np.mean(test_metric_list), np.std(test_metric_list)))
        else:
            self.logger.log_print("The test {} is: {}".format(self.params["test"]["metric_name"], test_metric_list[0]))
        
        
    def test(self, pth_path, files_list, vis_foloder):
        try:
            net_inference = NetInference(self.gpu_detector, self.params, net_class = self.net_class, pth_address = pth_path, post_process=self.data_postprocess)
        except AssertionError as e:
            print(e)
        except FileNotFoundError as e:
            print("The pth address is not right")
        
        test_vis_dir = Path(pth_path.parent, vis_foloder)
        test_metrics = 0.0
        for img_name in files_list:
            print("Analyzing {}".format(img_name))
            img    = load_data_from_disk(Path(self.params["data"]["dir"], "images", img_name))
            label  = load_data_from_disk(Path(self.params["data"]["dir"], "labels", img_name.replace(self.params["data"]["imgs_suffix"], "npy")))
            output = net_inference.forward_one_image(img)
            temp_metric = metrics.get_metric(self.params["test"]["metric_name"], output, label)
            self.logger.log_print("The {} of {} is {}".format(self.params["test"]["metric_name"], img_name, temp_metric))
            test_metrics += temp_metric
            
            if test_vis_dir is not None:
                test_vis_dir.mkdir(parents=True, exist_ok=True)
                vis_result(img, label, output, test_vis_dir, self.logger, 
                           description="test_" + img_name + "_", show_mask = self.params["test"]["show_mask"])
            
        return test_metrics / len(files_list)
    


class NetInference:
    """
        Net Inference for 2D image segmentation
    """
    def __init__(self, gpu_detector, params: Dict, net_class: Network, pth_address: Path = None, post_process = None):
        
        self.params = params
        
        # overlap-tile parameters
        self._input_size   = self.params["test"]["input_size"]
        self._overlap_size = self.params["test"]["overlap_size"]
        
        # batch_size in inference stage can improve the efficiency for overlap-tile strategy
        self._batch_size = self.params["test"]["batch_size"]
        
        # Test Time augmentation, aug_type = 0: no tta, 1: 4 variants, 2: 8 variants
        self._aug_type = self.params["test"]["aug_type"]
        
        self._mean = self.params["data"]["imgs_mean"]
        self._std  = self.params["data"]["imgs_std"]
        self._z_score_norm = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean = [self._mean], 
                         std  = [self._std])
        ])
        self._use_post_process = self.params["test"]["use_post_process"]
        
        self._gpu_detector = gpu_detector
        
        self._model = net_class.get_net(self.params["data"]["imgs_channels"], self.params["data"]["labels_class_num"])

        gpu_status, gpu_idx_in_use = self._gpu_detector.get_current_gpu()
        if gpu_status:
            self._model = nn.DataParallel(self._model, device_ids = gpu_idx_in_use).cuda()
            
        # model parameters setting
        self._pth_address = pth_address
        try:
            self.__load_pth()
        except (AssertionError, FileNotFoundError) as e:
            self._load_status = False
        else:
            self._load_status = True
        
        # post process
        self.post_process = post_process
    

    
    def forward_one_image(self, in_ori_img: np.ndarray) -> np.ndarray:
        """
            Forward one image, return segmentation result, some errers will be raised
        """
        assert self._load_status, "Please load parameters correctly"
        assert isinstance(in_ori_img, np.ndarray), "The input image must be numpy type"
        assert in_ori_img.ndim == 2,  "The iput image should be gray scale"
        assert isinstance(self._input_size, int) and self._input_size % 16 == 0, "The input size must be int and can be divided by 16"
        assert isinstance(self._overlap_size, int) and self._overlap_size > 0, "The overlap size must be int and greater than 0"
        assert self._input_size > (2 * self._overlap_size), "The input size {} must be greater than 2 times overlap size {}".format(self._input_size, self._overlap_size)
        assert self._aug_type in [0,1,2], "The parameter for test time augmentation must be a value in {0,1,2}"
        assert isinstance(self._batch_size, int) and self._batch_size > 0, "The batch size must be int and greater than 0"
        
        # For overlap tile strategy and tta, the sub imgs are in List Type
        overlap_tile = OverlapTile(crop_size = self._input_size, overlap_size = self._overlap_size)
        tta = TestTimeAug(aug_type = self._aug_type)
        
        self._model.eval()
        with torch.no_grad():
            # overlap-tile strategy, crop img into sub-imgs in List and revert after network inference
            in_imgs = overlap_tile.crop(in_ori_img)
            in_imgs = tta.aug_input(in_imgs)
            out_imgs = []  
            try:
                img_idx = 0
                h, w = in_imgs[0].shape
                while img_idx < len(in_imgs):
                    iter_batch_size = self._batch_size
                    if img_idx + iter_batch_size > len(in_imgs):
                        # if the len(in_imgs) can not be divided by batch_size, the batch size of last itereation should be 
                        # len(in_imgs) - img_idx
                        iter_batch_size = len(in_imgs) - img_idx
                    
                    in_tensor = torch.zeros((iter_batch_size, 1, h, w))
                    for b_idx in range(iter_batch_size):
                        in_tensor[b_idx, :, :, :] = self._z_score_norm(in_imgs[img_idx + b_idx])
                    
                    gpu_status, gpu_idx_in_use =  self._gpu_detector.get_current_gpu()
                    if gpu_status:
                        in_tensor = in_tensor.cuda(gpu_idx_in_use[0])                
                    out_tensor = self._model.forward(in_tensor)  # b,c,w,h c=2
                    out_tensor = F.softmax(out_tensor, dim = 1)
                    if gpu_status:
                        out_tensor = out_tensor.cpu()
                    
                    for b_idx in range(iter_batch_size):
                        # the out data is presented by List again
                        out_imgs.append(out_tensor[b_idx, :, :, :].numpy().transpose((1,2,0))) # c, h, w => h, w, c
                    img_idx = img_idx + iter_batch_size
                    torch.cuda.empty_cache()
            except RuntimeError as e: 
                raise e # GPU or CPU memory is out of use
            else:
                out_imgs = tta.merge_out(out_imgs) # merge different tta results.
                for out_idx in range(len(out_imgs)):
                    # transform type and post process
                    out_imgs[out_idx] = np.argmax(out_imgs[out_idx], axis=2).astype(np.uint8)
                out_img = overlap_tile.stitch(out_imgs)
                if self._use_post_process:
                    if self.post_process is None:
                        out_img = self.__post_process(out_img)
                    else: 
                        out_img = self.post_process(out_img)
        return out_img
    
    def __post_process(self, in_img: np.ndarray) -> np.ndarray:
        """
            Post processing method after network inference.
        """
        # skeletonization of result
        out_img = morphology.skeletonize(1 - in_img, method="lee") /255 # Lee Skeleton method
        # dialiation with 3 pixels
        out_img = 1 - morphology.dilation(out_img, morphology.square(3))
        # Todo: delete the noise region whose area smaller than threshold
        # Add this depend on the user's decision
        return out_img
        
    def __load_pth(self):
        """
            Load model parameters
        """
        assert self._pth_address, "Please set pth address"
        gpu_status, _ =  self._gpu_detector.get_current_gpu()
        try:
            if gpu_status: # load gpu parameters for gpu
                self._model.load_state_dict(torch.load(self._pth_address))
            else:          # load gpu parameters for cpu
                self._model.load_state_dict({k.replace('module.',''): v for k,v in torch.load(self._pth_address).items()})
        except FileNotFoundError as e:
            raise e
        
    def set_model(self, model):
        self._model = model
    
    def set_pth_address(self, pth_address: str):
        self._pth_address = pth_address
        try:
            self.__load_pth()
        except (AssertionError, FileNotFoundError) as e:
            self._load_status = False
            raise e
        else:
            self._load_status = True
    
    def set_input_size(self, input_size: int):
        assert isinstance(input_size, int) and input_size % 16 == 0, "The input size must be int and can be divided by 16"
        self._input_size = input_size

    def set_overlap_size(self, overlap_size: int):
        assert isinstance(overlap_size, int) and overlap_size > 0, "The overlap size must be int and greater than 0"
        self._overlap_size = overlap_size
    
    def set_batch_size(self, batch_size: int):
        assert isinstance(self._batch_size, int) and self._batch_size > 0, "The batch num must be int and greater than 0"
        self._batch_size = batch_size
    
    def set_aug_type(self, aug_type: int):
        assert self._aug_type in [0,1,2], "The parameter for test time augmentation must be a value in {0,1,2}"
        self._aug_type = aug_type
    
    def _set_use_post_process(self, use_post_process:bool):
        assert isinstance(use_post_process, bool), "The use_post_process must be bool"
        self._use_post_process = use_post_process
    
    def set_mean_and_std(self, mean:float, std:float):
        assert isinstance(mean, float) and mean > 0, "The mean must be float and greater than 0"
        assert isinstance(std, float) and std > 0,   "The std  must be float and greater than 0"
        self._mean = mean
        self._std = std


class OverlapTile:
    """
        This stategy is implementated from: Ma B, Ban X, Huang H, et al. Deep learning-based image segmentation
        for al-la alloy microscopic images[J]. Symmetry, 2018, 10(4): 107.
        For figure 4
        crop_size is the size of blue rectangle, which is equal to input_size
        roi_size is the size of yellow rectangle
    """

    def __init__(self, crop_size: int = 256, overlap_size: int = 32):
        self._crop_size = crop_size
        self._overlap_size = overlap_size
        self._roi_size = crop_size - 2 * overlap_size
        self._in_img_shape = None

    def crop(self, in_img: np.ndarray) -> List:
        """
            Crop in_img to sub-img List,
            hint: self._roi_size is used in crop and stitch stage, please check the paper careful when read this code
        """
        self._in_img_shape = in_img.shape
        # Pad image before cropping
        in_pad_img = np.pad(in_img, self._overlap_size, mode='symmetric')
        # calculate the number of cropping, which consider the influence of remainder
        h_pad_num = math.ceil(in_pad_img.shape[0] / self._roi_size)
        w_pad_num = math.ceil(in_pad_img.shape[1] / self._roi_size)
        # if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
        # if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
        in_crop_imgs = []
        for i in range(h_pad_num):
            # row analysis
            # overlap_crop, it is need to calculate the start of cropping
            start_h = i * self._roi_size
            end_h = start_h + self._crop_size

            # if there is some remainder result for cropping, change start_h and start_w
            if end_h > in_pad_img.shape[0]:
                start_h = in_pad_img.shape[0] - self._crop_size
                end_h = in_pad_img.shape[0]

            for j in range(w_pad_num):
                # column analysis
                start_w = j * self._roi_size
                end_w = start_w + self._crop_size

                if end_w > in_pad_img.shape[1]:
                    start_w = in_pad_img.shape[1] - self._crop_size
                    end_w = in_pad_img.shape[1]

                crop_img = in_pad_img[start_h: end_h, start_w: end_w]
                # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".\
                #      format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
                in_crop_imgs.append(crop_img)
                if end_w == in_pad_img.shape[1]:
                    break
            if end_h == in_pad_img.shape[0]:
                break
        return in_crop_imgs

    def stitch(self, out_crop_imgs: List) -> np.ndarray:
        """
            Stitch sub-img List to whole out img
        """
        out_img = np.zeros(self._in_img_shape)

        # calculate the number of cropping, which consider the influence of remainder
        # h_num = math.ceil(self._in_img_shape[0] / self._roi_size) # it is no need to use that
        w_num = math.ceil(self._in_img_shape[1] / self._roi_size)

        for img_idx, out_crop_img in enumerate(out_crop_imgs):
            roi_img = out_crop_img[self._overlap_size: self._overlap_size + self._roi_size,
                      self._overlap_size: self._overlap_size + self._roi_size]
            i = int(img_idx / w_num)
            j = int(img_idx - (i * w_num))
            start_h = int(i * self._roi_size);
            end_h = start_h + self._roi_size
            start_w = int(j * self._roi_size);
            end_w = start_w + self._roi_size

            if end_h > self._in_img_shape[0]:
                start_h = self._in_img_shape[0] - self._roi_size
                end_h = self._in_img_shape[0]
            if end_w > self._in_img_shape[1]:
                start_w = self._in_img_shape[1] - self._roi_size
                end_w = self._in_img_shape[1]
                # print("stitching: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}"\
            # .format(i, start_h, start_w, j, end_h, end_w))
            out_img[start_h: end_h, start_w: end_w] = roi_img
        return out_img


class TestTimeAug:
    """
        Test Time Augmentation, you can add your own function in this class
    """
    def __init__(self, aug_type: int = 0):
        self._aug_type = aug_type   # 0: no tta; 1: 4 variant(rotation), 2: eight variant(rotation and flip)
    
    def aug_input(self, in_imgs: List) -> List:
        """
            Aug sub-imgs List and return a new List
        """
        in_aug_imgs = []
        for in_img in in_imgs:
            if self._aug_type >= 0:    
                in_aug_imgs.append(in_img.copy())
            if self._aug_type >= 1:                            # rotation 4 variants for one input
                in_aug_imgs.append(np.rot90(in_img, 1).copy()) # if there is no copy, it will introduce error in network inference
                in_aug_imgs.append(np.rot90(in_img, 2).copy())
                in_aug_imgs.append(np.rot90(in_img, 3).copy())
            if self._aug_type >= 2:                            # rotation and flip 8 variants for one input
                in_aug_imgs.append(np.flipud(in_img).copy())   # upper and down flip
                in_aug_imgs.append(np.fliplr(in_img).copy())   # left and right flip
                in_aug_imgs.append(np.flipud(np.rot90(in_img, 3)).copy())  # upper and down flip for 270 rotation
                in_aug_imgs.append(np.fliplr(np.rot90(in_img, 3)).copy())  # left and right flip for 270 rotation
        return in_aug_imgs
                
    def merge_out(self, out_aug_imgs: List) -> List:
        """
            Merge aug List(Mean) and return sub-imgs List
        """
        out_imgs = []
        tta_num = self._aug_type * 4 # for aug_type = 1, 4 variants, for aug_type = 2, 8 variants.
        idx = 0
        while idx < len(out_aug_imgs):
            if self._aug_type >= 0:    
                sum_imgs = out_aug_imgs[idx].copy() 
            if self._aug_type >= 1:                
                sum_imgs = sum_imgs + np.rot90(out_aug_imgs[idx + 1], -1)
                sum_imgs = sum_imgs + np.rot90(out_aug_imgs[idx + 2], -2)
                sum_imgs = sum_imgs + np.rot90(out_aug_imgs[idx + 3], -3)
            if self._aug_type >= 2: 
                sum_imgs = sum_imgs + np.flipud(out_aug_imgs[idx + 4])
                sum_imgs = sum_imgs + np.fliplr(out_aug_imgs[idx + 5])
                sum_imgs = sum_imgs + np.rot90(np.flipud(out_aug_imgs[idx + 6]), -3)
                sum_imgs = sum_imgs + np.rot90(np.fliplr(out_aug_imgs[idx + 7]), -3)
            if self._aug_type == 0:
                out_imgs.append(sum_imgs)
                idx = idx + 1
            elif self._aug_type >= 1:
                out_imgs.append(sum_imgs / tta_num)
                idx = idx + tta_num
        return out_imgs