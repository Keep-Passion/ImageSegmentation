import os
import random
import numpy as np
import torch
import math
from libtiff import TIFF
from pathlib import Path
from skimage import io
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as tr
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Set, Tuple
from IPython.display import clear_output
from skimage import morphology

from model.nets.unet import UNet
from model.nets.resunet import ResUNet
from model.nets.attunet import AttUNet
from model.nets.nestedunet import NestedUNet


def setup_seed(seed: int) -> None:
    """
        Set random seed to make experiments repeatable
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # implement same config in cpu and gpu
    torch.backends.cudnn.benchmark = True


def count_mean_and_std(img_dir: Path, img_suffix: str) -> Tuple: # (mean, std, num) (0.495770570343616, 0.16637932545720774, 22) for ISBI 2012 
    """
        Calculate mean and std of data of original image
    """
    assert img_dir.is_dir(), "The input is not a dir"
    mean, std, num = 0, 0, 0

    imgs_path = img_dir.glob("*.{}".format(img_suffix))
    
    for img_path in imgs_path:
        num += 1
        img = io.imread(str(img_path), 0) * 1.0 / 255
        assert np.max(np.unique(img)) <= 1, "The img value should lower than 1 when calculate mean and std"
        mean += np.mean(img)
        std += np.std(img)
    mean /= num
    std /= num
    return mean, std, num
    

def load_img(img_path: Path) -> np.ndarray:
    """
        Load images or npy files, you can add your own type of file in here
    """
    if img_path.suffix == '.npy':
        img = np.load(img_path)
    else:
        img = io.imread(str(img_path))
    return img


class Logger():
    """
        Log and print the information during experiment
    """
    def __init__(self, is_out_log_file=True, file_address=None) -> None:
        self.is_out_log_file = is_out_log_file
        self.file_address = file_address
    
    def log_print(self, content) -> None:
        print(content)
        if self.is_out_log_file:
            f = open(str(self.file_address), "a")
            f.write(content)
            f.write("\n")
            f.close()

            
def adjust_learning_rate(optimizer, epoch, learning_rate, dacay_ratio: float = 0.8, decay_num: int = 10, lower_limit:float = 1e-6) -> None:
    """
        Sets the learning rate to the initial LR decayed by dacay_ratio every decay_num epochs until the lower_limit
    """
    lr = learning_rate * (dacay_ratio ** (epoch // decay_num))
    if not lr < lower_limit:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# checking function
def check_result(epoch, img, label, output, weight, checkpoint_path, logger, description="val_"): # check the output of dataset
    """
        Check visualization of result during training
    """
    logger.log_print("Image size is {},    min is {}, max is {}".format(img.shape, np.amin(img), np.amax(img)))
    logger.log_print("Label size is {},    min is {}, max is {}".format(label.shape, np.amin(label), np.amax(label)))
    logger.log_print("Output size is {},   min is {}, max is {}".format(output.shape, np.amin(output), np.amax(output)))
    logger.log_print("Weight-0 size is {}, min is {}, max is {}".format(weight[:, :, 0].shape, np.amin(weight[:, :, 0]), np.amax(weight[:, :, 0])))
    logger.log_print("Weight-1 size is {}, min is {}, max is {}".format(weight[:, :, 1].shape, np.amin(weight[:, :, 1]), np.amax(weight[:, :, 1])))
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 5, 1), plt.imshow(img,    cmap="gray"), plt.title("img"), plt.axis("off")
    plt.subplot(1, 5, 2), plt.imshow(label,  cmap="gray"), plt.title("label"), plt.axis("off")
    plt.subplot(1, 5, 3), plt.imshow(output, cmap="gray"), plt.title("output"), plt.axis("off")
    plt.subplot(1, 5, 4), plt.imshow(weight[:, :, 0], cmap="plasma"), plt.title("weight-0"), plt.axis("off")
    plt.subplot(1, 5, 5), plt.imshow(weight[:, :, 1], cmap="plasma"), plt.title("weight-1"), plt.axis("off")
    plt.savefig(str(Path(checkpoint_path, description + str(epoch).zfill(3) + "_result.png")))
    plt.show()


def plot(epoch, train_value_list, val_value_list, checkpoint_path, find_min_value = True, curve_name = "loss", ignore_epoch: int = 2) -> None:
    """
        Plot training cure during training, and draw target epoch in the figure
    """
    clear_output(True)
    plt.figure()
    target_value = 0
    target_func = 'None'
    if find_min_value and len(val_value_list) > ignore_epoch:
        target_value = min(val_value_list[ignore_epoch:])
        target_func = 'min'
    elif find_min_value is False and len(val_value_list) > ignore_epoch:
        target_value = max(val_value_list[ignore_epoch:])
        target_func = 'max'
    title_name = 'Epoch {}. train ' +  curve_name + ': {:.4f}. val ' + curve_name + ': {:.4f}. ' + ' val' + target_func + ' ' + curve_name + ': {:.4f}. '
    plt.title(title_name.format(epoch, train_value_list[-1], val_value_list[-1], target_value))
    plt.plot(train_value_list, color="r", label="train " + curve_name)
    plt.plot(val_value_list,   color="b", label="val "  + curve_name)
    if len(val_value_list) > ignore_epoch:
        plt.axvline(x = val_value_list.index(target_value), ls='-',c='green')
        plt.legend(loc = 'best')
    plt.savefig(str(Path(checkpoint_path,  curve_name + '_curve.png')))
    plt.show()

    
def uniq(a: Tensor) -> Set:
    """ Get the set of unique value in tensor """
    return set(torch.unique(a.cpu()).numpy())


def is_sset(a: Tensor, sub: Iterable) -> bool:
    """ 
        Judge the unique value of a is the subset of sub 
        is_set(a, [0,1])
    """  
    return uniq(a).issubset(sub)


class GpuDetector():
    """
        Gpu detector to address the problems related to the gpu.
    """
    def __init__(self):
        self._gpu_number = torch.cuda.device_count() # if there is np gpu, return 0
        self._gpu_idx_in_use = -1                    # which gpu to use 
        self._gpu_is_available = False
        for gpu_idx in range(self._gpu_number):
            try:
                self.__check_gpu_status(gpu_idx)
            except AssertionError as e:
                pass
            else:
                self._gpu_is_available = True
                self._gpu_idx_in_use = gpu_idx
                break
        # Todo: the multi-gpus inference
    
    def set_gpu_device(self, gpu_idx: int) -> Tuple:
        """
            Setting certain gpu advice to use
        """
        try:
            self.__check_gpu_status(gpu_idx)
        except AssertionError as e:  # if there is error, raise it to upper call
            raise e
        else:
            self._gpu_is_available = True
            self._gpu_idx_in_use = gpu_idx
        # Todo: the multi-gpus inference 
    
    def get_current_gpu(self) -> Tuple:
        """
            Get current gpu status and gpu device
        """
        # return self._gpu_is_available, self._gpu_idx_in_use
        return False, self._gpu_idx_in_use

    def __check_gpu_status(self, gpu_idx: int):
        """
            Check gpu status, some errors will return by AssertionError
        """
        assert torch.cuda.is_available(), "This computer has no gpu"
        assert isinstance(gpu_idx, int) and gpu_idx >= 0, "The gpu index must be int and greater than 0"
        assert gpu_idx <= (self._gpu_number - 1), "There is only {} gpus and has no gpu with idx {}".format(self._gpu_number, gpu_idx)
        assert torch.cuda.get_device_capability(gpu_idx)[0] >= 2.0, "The device capability of gpu {} with name {} is lower than 2.0".format(gpu_idx, torch.cuda.get_device_name(gpu_idx))
        # Todo: memory size and useable memory size detection,
        # however this can be handle in RuntimeError in inference


class NetInference():
    """
        Net Inference for 2D image segmentation
    """
    def __init__(self, gpu_detector, input_size:int=512, overlap_size:int=32, batch_size:int=1, 
                 aug_type:int=0, mean:float=0.9410404628082503, std:float=0.12481161024777744, 
                 model = UNet, in_channel: int = 1, out_channel:int = 2, pth_address:str=None, use_post_process:bool=True):
        # overlap-tile parameters
        self._input_size = input_size
        self._overlap_size = overlap_size
        
        # batch_size in inference stage can improve the efficiency for overlap-tile strategy
        self._batch_size = batch_size
        
        # Test Time augmentation, aug_type = 0: no tta, 1: 4 variants, 2: 8 variants
        self._aug_type = aug_type
        
        self._mean = mean
        self._std = std
        self._z_score_norm = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean = [self._mean], 
                         std =  [self._std])
        ])
        self._use_post_process = use_post_process
        
        self._gpu_detector = gpu_detector
        self._model = model(in_channel, out_channel)
        gpu_status, gpu_idx_in_use = self._gpu_detector.get_current_gpu()
        if gpu_status:
            self._model = nn.DataParallel(self._model).cuda(gpu_idx_in_use)
            
        # model parameters setting
        self._pth_address = pth_address
        try:
            self.__load_pth()
        except (AssertionError, FileNotFoundError) as e:
            self._load_status = False
        else:
            self._load_status = True
    
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
                        in_tensor = in_tensor.cuda(gpu_idx_in_use)                
                    out_tensor = self._model.forward(in_tensor) # b,c,w,h c=2
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
                    out_img = self.__post_process(out_img)
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


class OverlapTile():
    """
        This stategy is implementated from: Ma B, Ban X, Huang H, et al. Deep learning-based image segmentation for al-la alloy microscopic images[J]. Symmetry, 2018, 10(4): 107.
        For figure 4
        crop_size is the size of blue rectangle, which is equal to input_size
        roi_size is the size of yellow rectangle
    """
    def __init__(self, crop_size:int=256, overlap_size:int=32):
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
        h_pad_num = math.ceil(in_pad_img.shape[0] / (self._roi_size))
        w_pad_num = math.ceil(in_pad_img.shape[1] / (self._roi_size))
        # if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
        # if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
        in_crop_imgs = []
        for i in range(h_pad_num):
            for j in range(w_pad_num):
                # overlap_crop, it is need to calculate the start of cropping
                start_h = i * self._roi_size; end_h = start_h + self._crop_size 
                start_w = j * self._roi_size; end_w = start_w + self._crop_size
                
                # if there is some remainder result for cropping, change start_h and start_w
                if end_h > in_pad_img.shape[0]:
                    start_h = in_pad_img.shape[0] - self._crop_size
                    end_h = in_pad_img.shape[0]
                if end_w > in_pad_img.shape[1]:
                    start_w = in_pad_img.shape[1] - self._crop_size
                    end_w = in_pad_img.shape[1]
                
                crop_img = in_pad_img[start_h: end_h, start_w: end_w]
                # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
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
        h_num = math.ceil(self._in_img_shape[0] / self._roi_size)
        w_num = math.ceil(self._in_img_shape[1] / self._roi_size)
        
        for img_idx, out_crop_img in enumerate(out_crop_imgs):
            roi_img = out_crop_img[self._overlap_size: self._overlap_size + self._roi_size, 
                                   self._overlap_size: self._overlap_size + self._roi_size]
            i = int(img_idx / w_num)
            j = int(img_idx - (i * w_num))
            start_h = int(i * self._roi_size); end_h = start_h + self._roi_size
            start_w = int(j * self._roi_size); end_w = start_w + self._roi_size
            
            if end_h > self._in_img_shape[0]:
                start_h = self._in_img_shape[0] - self._roi_size
                end_h = self._in_img_shape[0]
            if end_w > self._in_img_shape[1]:
                start_w = self._in_img_shape[1] - self._roi_size
                end_w = self._in_img_shape[1]    
            # print("stitching: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}".format(i, start_h, start_w, j, end_h, end_w))    
            out_img[start_h: end_h, start_w: end_w] = roi_img
        return out_img


class TestTimeAug():
    """
        Test Time Augmentation, you can add your own function in this class
    """
    def __init__(self, aug_type: int = 0):
        self._aug_type = aug_type # 0: no tta; 1: 4 variant(rotation), 2: eight variant(rotation and flip)
    
    def aug_input(self, in_imgs:List) -> List:
        """
            Aug sub-imgs List and return a new List
        """
        in_aug_imgs = []
        for in_img in in_imgs:
            if self._aug_type >= 0:    
                in_aug_imgs.append(in_img.copy())
            if self._aug_type >= 1: # rotation 4 variants for one input
                in_aug_imgs.append(np.rot90(in_img, 1).copy()) # if there is no copy, it will introduce error in network inference
                in_aug_imgs.append(np.rot90(in_img, 2).copy())
                in_aug_imgs.append(np.rot90(in_img, 3).copy())
            if self._aug_type >= 2: # rotation and flip 8 variants for one input
                in_aug_imgs.append(np.flipud(in_img).copy()) # upper and down flip
                in_aug_imgs.append(np.fliplr(in_img).copy()) # left and right flip
                in_aug_imgs.append(np.flipud(np.rot90(in_img, 3)).copy()) # upper and down flip for 270 rotation
                in_aug_imgs.append(np.fliplr(np.rot90(in_img, 3)).copy()) # left and right flip for 270 rotation
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
    