import os
import zipfile
import random
import numpy as np
import torch
from pathlib import Path
from skimage import io
from torch import Tensor
from typing import Callable, Iterable, List, Set, Tuple, Dict
from ruamel import yaml


def setup_seed(seed: int = 2021) -> None:
    """
        Set random seed to make experiments repeatable
        * Note: the repeatable of network training should futher studied
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True    # implement same config in cpu and gpu
    torch.backends.cudnn.benchmark = True


def make_zip(source_dir: Path, output_filename: str) -> None:
    """
        zip files in source_dir to 'output_filename'(.zip)
        :param source_dir: the directory of files needs to be zipped
        :param output_filename: should end with ".zip"
    """
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dir_names, file_names in os.walk(source_dir):
        for file_name in file_names:
            path_file = os.path.join(parent, file_name)
            arc_name = path_file[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(path_file, arc_name)
    zipf.close()


def unzip_file(source_path: Path, output_dir: Path) -> None:
    """
        unzip files
        :param source_path:  the path of source zip file
        :param output_dir:   the directory that will save unzip files
    """
    zipf = zipfile.ZipFile(str(source_path))
    zipf.extractall(str(output_dir))
    zipf.close()


def load_data(data_list: List, data_paths: Path, data_index: int) -> np.ndarray:
    """
        load data from memory or from disk
        :param data_list: None or List, if data is None, it will load data from disk, otherwise it will read data from pre-storage list
        :param data_paths: the list of data path in disk, each element of it is a data path
        :param data_index: use to identify the data for dataloader 
        :return: numpy data 
    """
    if data_list is None:
        return load_data_from_disk(data_paths[data_index])
    else:
        return data_list[data_index]
    

def load_data_from_disk(data_path: Path) -> np.ndarray:
    """
        Load images or npy files from disk, you can add your own type of file in here
        :param data_path: the path of data in disk
        :return: numpy data 
    """
    if data_path.suffix == ".npy":
        data = np.load(str(data_path))
    else:
        data = io.imread(str(data_path))
    return data


def load_data_to_memory(data_paths: List) -> List:
    """
        Load data to list in cpu memory according before network training
        :param data_paths: the list of data path in disk, each element of it is a data path
        :return: List, which storage the data in memory 
    """
    data_list = []
    for data_path in data_paths:
        if data_path.suffix == ".npy":
            data_list.append(np.load(str(data_path)))
        else:
            data_list.append(io.imread(str(data_path)))
    return data_list


def load_tensor_to_cgpu(in_tensor: Tensor, params: Dict, gpu_ids: List) -> Tensor:
    """
        Load tensor to cpu/gpu before feed to network in training stage
        :param in_tensor: input tesnor which need to be analysed
        :param params: the dict variable which mantain the experiment parameters
        :return: output tensor to feed the network
    """
    if torch.cuda.is_available() and params["gpu"]["use_gpu"]:
        return in_tensor.cuda(gpu_ids[0])
    else:
        return in_tensor

    
def count_mean_and_std(img_dir: Path, img_suffix: str) -> Tuple:  
    """
        Calculate mean and std of data of original image
        :param img_dir: the directory of image in disk
        :param img_suffix: the img suffix in img_dir
        :return: (mean, std, num) (0.495770570343616, 0.16637932545720774, 22) for ISBI 2012
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

            
def adjust_learning_rate(optimizer, epoch, learning_rate, dacay_ratio: float = 0.8, decay_num: int = 10, lower_limit:float = 1e-6) -> None:
    """
        Deprecated
        Sets the learning rate to the initial LR decayed by dacay_ratio every decay_num epochs until the lower_limit
    """
    lr = learning_rate * (dacay_ratio ** (epoch // decay_num))
    if not lr < lower_limit:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def uniq(a: Tensor) -> Set:
    """
        Get the set of unique value in tensor
    """
    return set(torch.unique(a.cpu()).numpy())


def is_sset(a: Tensor, sub: Iterable) -> bool:
    """ 
        Judge the unique value of a is the subset of sub 
    """  
    return uniq(a).issubset(sub)


class GpuDetector:
    """
        Gpu detector to address the problems related to the gpu.
    """
    def __init__(self):
        self._gpu_number = torch.cuda.device_count() # if there is np gpu, return 0
        self._gpu_ids_in_use = []                    # which gpus to use 
        self._gpu_is_available = False
    
    def set_gpu_device(self, gpu_list: List):
        """
            Setting certain gpus (gpu_list) advice to use
        """
        gpu_actual_work = []
        for gpu_idx in gpu_list:
            try:
                self.__check_gpu_status(gpu_idx)
            except AssertionError as e:  # if there is error, raise it to upper call
                print(e)
            else:
                self._gpu_is_available = True
                gpu_actual_work.append(gpu_idx)
        if gpu_actual_work != gpu_list:
            print("The system only have gpu devices {} can work, not the required devices {}".format(str(gpu_actual_work), str(gpu_list)))
        self._gpu_ids_in_use = gpu_actual_work
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_ids_in_use)
    
    def get_current_gpu(self) -> Tuple:
        """
            Get current gpu status and gpu device
        """
        return self._gpu_is_available, self._gpu_ids_in_use
        # return False, self._gpu_idx_in_use

    def __check_gpu_status(self, gpu_idx: int):
        """
            Check gpu (gpu_idx) status, some errors will return by AssertionError
        """
        assert torch.cuda.is_available(), "This computer has no gpu"
        assert isinstance(gpu_idx, int) and gpu_idx >= 0, "The gpu index must be int and greater than 0"
        assert gpu_idx <= (self._gpu_number - 1), "There is only {} gpus and has no gpu with idx {}".format(self._gpu_number, gpu_idx)
        assert torch.cuda.get_device_capability(gpu_idx)[0] >= 2.0, "The device capability of gpu {} with name {} is lower than 2.0".format(gpu_idx, torch.cuda.get_device_name(gpu_idx))



class Logger:
    """
        Log and print the information during experiment
    """
    def __init__(self, is_out_log_file = True, file_address = None) -> None:
        """
            :param is_out_log_file:  wehter or not to save log file to the disk
            :param file_address: the path of log file in disk
        """
        self.is_out_log_file = is_out_log_file
        self.file_address = file_address
    
    def log_print(self, content) -> None:
        """
            :param content: the content to show in the screen and log in the file (add to the last)
        """
        print(content)
        if self.is_out_log_file:
            f = open(str(self.file_address), "a")
            f.write(content)
            f.write("\n")
            f.close()


class ParamsYml:
    """
        Load and save the parameters of settings
    """
    def __init__(self):
        # default_params which can be load without actual yaml file
        self.default_params = {
            # data parameters
            "data":{
                "name": "isbi2012",
                "dir": None,
                "imgs_num":  30,
                "imgs_mean": 0.49475772084753505,
                "imgs_std":  0.16781602590161787,
                "imgs_channels": 1,
                "imgs_depth": None,
                "imgs_suffix": "png",
                "labels_class_num": 2,
                "weights_name": "bcw",
                "use_kfold": False,
                "kf_num": 5,
                "val_rate": 0.1,
                "is_load_data_to_cpu_mem":True,
            },
            # gpu parametrs
            "gpu":{
                "use_gpu": True,
                "cuda_devices": [0],
            },
            # training parameters
            "train":{
                "seed": 2020,
                "model_name": "unet",       # "unet", "resunet", "attunet", "nestedunet"
                "batch_size":10,
                "crop_size": 512,
                "num_worker":32,            # 0 or cpu kernels * 2 or gpu devices * 4, note it may cause unrepeatable
                "pin_memory": True,
                "lr": 1e-4,
                "optimizer": None,
                # ['stepLR', 10, 0.8]
                "scheduler": ["stepLR", 10, 0.8],
                "epochs": 50,
                "loss_name": "bcw",         # "bcw", "uw", "dice"
                # "rand_crop", "rand_rotate", "rand_vflip", "rand_hflip", "rand_zflip"
                "train_augment": ["rand_crop", "rand_rotate", "rand_vflip", "rand_hflip"],
            },
            # val parameters
            "val":{
                "batch_size": 10,
                "metric_name": "loss",      # "loss", "dice", "vi"
                "is_show_first_img": True,
                "show_interval": 1,
                "ignore_show_epoch_num": 2, 
            },
            # inference parameters
            "test":{
                "input_size": 512,
                "overlap_size": 16,
                "batch_size": 2,
                "aug_type": 0,              # Test Time augmentation, aug_type = 0: no tta, 1: 4 variants, 2: 8 variants
                "use_post_process": False,
                "metric_name": "vi",
                "show_mask": False,
            }          
        }
    
    
    def load_params(self, in_yaml_path: Path = None) -> Dict:
        """
            load parameters to the memory
            :param in_yaml_path: None or the path of experiment parameters, if it is None, it will load default_params
            :return: Dict of experiment parameters
        """
        out_params = None
        if in_yaml_path is None:
            out_params = self.default_params
        else:
            with open(str(in_yaml_path), 'r') as f:
                out_params = yaml.load(f.read(), Loader=yaml.Loader) 
        return out_params
    
    
    def save_params(self, out_yaml_path: Path, params: Dict = None) -> None:
        """
            save params to the disk. Generally, it needs to save parameters in each experiment directory
            :param out_yaml_path: the path of experiment parameters needs to be saved in disk
            :param params: the dict variable which mantain the experiment parameters
        """
        with open(str(out_yaml_path), 'w', encoding="utf-8") as f:
            if params is not None:
                f.write(yaml.dump(params, Dumper=yaml.RoundTripDumper)) 
            else:
                f.write(yaml.dump(self.default_params, Dumper=yaml.RoundTripDumper)) 

                
def set_experiment_name(params: Dict, custom_desc: str) -> str:
    """
        Set experiment name according to params, you can add your own style in here
        :param params: the dict variable which mantain the experiment parameters
        :param custom_desc: the custom description add to the last of experiment name
        :return: the string of name
    """
    experiment_name = 'seg_' + params["data"]["name"] + '_' + params["train"]["model_name"] + \
                      '_loss_' + params["train"]["loss_name"]
    if params["data"]["use_kfold"]:
        experiment_name += '_' + str(params["data"]["kf_num"]) + "_fold"
    experiment_name += '_' + custom_desc
    return experiment_name
