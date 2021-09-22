import os
from pathlib import Path
from IPython.display import clear_output
import matplotlib.pyplot as plt

import cv2, time
import random
import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from sklearn.model_selection import KFold
import pandas as pd

import metrics
from data_preprocess import *
from utils import *
from losser import *
from model.nets.unet import UNet
from model.nets.resunet import ResUNet
from model.nets.attunet import AttUNet
from model.nets.nestedunet import NestedUNet

from dataloader import SegDataset
from data_preprocess import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SegTrainer:
    """
        Segmentation class
    """

    def __init__(self) -> None:

        # model parameters
        self.model_name = "unet"  # unet resunet attunet, nestedunet
        self.model_class = self.__check_model()
        self.model = None
        self.loss_name = "dice"  # 'bcw', 'uw', 'dice'
        self.input_channels = 1
        self.labels_class = 2

        # data parameters
        self.dataset_name = "isbi2012"
        self.data_experiment_path = None
        self.imgs_path = None
        self.labels_path = None
        self.weights_path = None
        self.files_name = None

        self.imgs_num = 30
        self.imgs_mean = 0.49475772084753505
        self.imgs_std = 0.16781602590161787
        self.z_score_norm = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=[self.imgs_mean],
                         std=[self.imgs_std])
        ])
        self.imgs_suffix = "png"
        self.zfill_num = 3

        # training parameters
        self.seed_num = 2020
        self.use_kfold = False
        self.kf_num = 5
        self.val_rate = 0.1
        self.val_metric_name = "loss"  # 'loss', 'dice', 'vi'
        self.is_min_val_metirc = self.__check_min_val_metric()

        self.batch_size = 10  # at least 2
        self.crop_size = 256  # 128,256,512
        self.learning_rate = 1e-4
        self.epochs = 50  # 10

        self.use_augment = [True, True, True]  # In training, rand_rotation, rand_vertical_flip, rand_horizontal_flip
        self.no_augment = [False, False, False]  # In Validate and test

        self.optimizer = None
        self.scheduler = None
        self.ignore_epoch_num = 2
        self.best_model_path = None

        # loss parameters
        self.losser = Losser()

        # gpu parameters
        self.gpu_detector = GpuDetector()
        try:
            self.gpu_detector.set_gpu_device(0)
        except AssertionError as e:
            print(e)

        # inference parameters
        self.test_input_size = 256  # overlap-tile parameters
        self.test_overlap_size = 16  # overlap-tile parameters

        self.test_batch_size = 2  # batch_size in inference stage can improve the efficiency for overlap-tile strategy
        self.test_aug_type = 0  # Test Time augmentation, aug_type = 0: no tta, 1: 4 variants, 2: 8 variants
        self.test_use_post_process = False  # whether or not use post processing in test time
        self.test_metric_name = "dice"  # metirc for test data, "dice" or "vi"

    def __check_model(self):
        models_name_list = ["unet", "resunet", "attunet", "nestedunet"]
        models_class_list = [UNet, ResUNet, AttUNet, NestedUNet]
        assert self.model_name in models_name_list, "{} is not support in this code".format(self.model_name)
        model_class = models_class_list[models_name_list.index(self.model_name)]
        return model_class

    def __check_min_val_metric(self) -> bool:
        if self.val_metric_name in ["loss", "vi"]:
            return True
        else:
            return False

    def data_preprocess(self, data_bk_path: Path, data_experiment_path: Path) -> None:
        print("Data Preprocess ... ")
        data_preprocess(self.dataset_name, data_bk_path, data_experiment_path, zfill_num=self.zfill_num,
                        suffix=self.imgs_suffix)  # you can change this
        print('Generating data in {}'.format(str(data_experiment_path)))

        # count mean, std and image num
        # for ISBI 2012, 0.49475772084753505 0.16781602590161787 30
        self.imgs_mean, self.imgs_std, self.imgs_num = count_mean_and_std(self.imgs_path, self.imgs_suffix)
        # mean, std, num = 0.49475772084753505, 0.16781602590161787, 30
        print("mean = {}, std ={} and image num = {}".format(self.imgs_mean, self.imgs_std, self.imgs_num))
        self.z_score_norm = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=[self.imgs_mean],
                         std=[self.imgs_std])
        ])

        # Calculate weight map
        # Balanced Class Weight, bcw, Unet Weight, uw
        weights_func = ["bcw", "uw"]
        prepare_weight_map(self.labels_path, data_experiment_path, weights_func, self.labels_class)
        print("Generated weight map in {}".format(str(data_experiment_path)))

        print("Data Preprocess has done")

    def start_training(self, data_bk_path: Path, data_experiment_path: Path, experiment_path: Path,
                       use_preprocess: bool = False) -> None:

        self.data_experiment_path = data_experiment_path
        self.imgs_path = Path(data_experiment_path, "images")
        self.labels_path = Path(data_experiment_path, "labels")
        if self.loss_name in ["bcw", "uw"]:
            self.weights_path = Path(data_experiment_path, self.loss_name)
        else:
            self.weights_path = Path(data_experiment_path, "bcw")
        if use_preprocess:
            self.data_preprocess(data_bk_path, data_experiment_path)
        self.files_name = [str(item.name) for item in list(self.imgs_path.glob("*.{}".format(self.imgs_suffix)))]

        kf = KFold(n_splits=self.kf_num, shuffle=True, random_state=self.seed_num)

        test_metrics = []

        experiment_name = 'seg_' + self.dataset_name + '_' + self.model_name + '_train_' + self.loss_name + \
                          '_val_' + self.val_metric_name + '_test_' + self.test_metric_name
        if self.use_kfold:
            experiment_name += '_' + str(self.kf_num) + '_fold'

        global_parameters_path = Path(experiment_path, experiment_name)
        global_parameters_path.mkdir(parents=True, exist_ok=True)
        logger = Logger(True, Path(global_parameters_path, "log.txt"))

        logger.log_print("Experiment: {}".format(experiment_name))

        # K-fold Cross Validation
        # files_name = [‘001.png’, ...]
        for kf_idx, (train_files, test_files) in enumerate(kf.split(self.files_name)):
            # Dataset split
            # we use k-fold cv to produce train set and test set, and then sample some data (10%)
            # from train set as val set (sampling without replacement)

            val_files = random.sample(list(train_files), int(len(train_files) * self.val_rate) + 1)
            train_files = list(set(train_files).difference(set(val_files)))
            train_files.sort();
            val_files.sort()

            train_files = [self.files_name[item] for item in train_files]
            val_files = [self.files_name[item] for item in val_files]
            test_files = [self.files_name[item] for item in test_files]

            if self.use_kfold:
                temp_parameters_path = Path(global_parameters_path, str(self.kf_num) + '_fold_' + str(kf_idx))
                temp_parameters_path.mkdir(parents=True, exist_ok=True)
                logger.log_print("Cross Validation: {}".format(str(self.kf_num) + '_fold_' + str(kf_idx)))
            else:
                temp_parameters_path = global_parameters_path
                if kf_idx >= 1:
                    break

            train_dataset = SegDataset(self.data_experiment_path, train_files, self.weights_path,
                                       use_augment=self.use_augment, crop_size=self.crop_size, depth=None,
                                       norm_transform=self.z_score_norm)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            train_one_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            val_dataset = SegDataset(self.data_experiment_path, val_files, self.weights_path,
                                     use_augment=self.no_augment, crop_size=self.crop_size, depth=None,
                                     norm_transform=self.z_score_norm)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            self.model = self.model_class(num_channels=self.input_channels, num_classes=self.labels_class)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
            if torch.cuda.is_available():
                self.model = nn.DataParallel(self.model).cuda()

            train_metric_list = []
            val_metric_list = []

            val_baseline = 0
            if self.is_min_val_metirc:
                val_baseline = 10000

            val_best_epoch = 0

            # Training
            st_total = time.time()
            logger.log_print("Training:")
            for i in range(1, self.epochs + 1):
                print("Experiment name: " + experiment_name)
                st = time.time()
                self.__train(train_loader)
                train_metric = self.__val(i, train_one_loader, temp_parameters_path, logger, check_first_img=False)
                val_metric = self.__val(i, val_loader, temp_parameters_path, logger, check_first_img=True)

                train_metric_list.append(train_metric)
                val_metric_list.append(val_metric)

                logger.log_print(
                    "Epoch {}: train_{} {:.4f}; val_{} {:.4f} \n".format(i, self.val_metric_name, train_metric_list[-1],
                                                                         self.val_metric_name, val_metric_list[-1]))

                plot(i, train_metric_list, val_metric_list, temp_parameters_path, self.is_min_val_metirc,
                     curve_name=self.val_metric_name, ignore_epoch=self.ignore_epoch_num)

                if (i > self.ignore_epoch_num) and ((self.is_min_val_metirc and val_metric < val_baseline) or (
                        self.is_min_val_metirc is False and val_metric > val_baseline)):
                    val_baseline = val_metric
                    val_best_epoch = i
                    torch.save(self.model.state_dict(), str(Path(temp_parameters_path, "best_model_state.pth")))
                ed = time.time()
                logger.log_print("Epoch Duration: {}'s".format(ed - st))
            ed_total = time.time()
            logger.log_print("Total duration is: {}'s".format(ed_total - st_total))
            logger.log_print("The best epoch is at: {} th epoch".format(val_best_epoch))
            logger.log_print("Train {} list is: {}".format(self.val_metric_name, train_metric_list))
            logger.log_print("Val   {} list is: {}".format(self.val_metric_name, val_metric_list))

            self.best_model_path = Path(temp_parameters_path, "best_model_state.pth")

            test_vis_path = Path(temp_parameters_path, "test_vis")
            torch.cuda.empty_cache()
            test_metrics.append(
                self.test(self.imgs_path, self.labels_path, test_files, self.best_model_path, test_vis_path))
            logger.log_print("test_metrics is {}".format(test_metrics))
        logger.log_print("Experiment is {}".format(experiment_name))
        logger.log_print("Metric is {} in test time".format(self.test_metric_name))
        if self.use_kfold:
            logger.log_print(
                "For cross validation, mean is: {}, std is {}".format(np.mean(test_metrics), np.std(test_metrics)))
        else:
            logger.log_print("The test value is: {}".format(test_metrics[0]))

    def __train(self, dataloader):
        self.model.train()
        for sample in dataloader:  # 2D [B, C, H, W]
            if torch.cuda.is_available():
                img, label = sample['img'].cuda(), sample['label'].cuda()
                if self.weights_path is not None:
                    weight = sample['weight'].cuda()
            else:
                img, label = sample['img'], sample['label']
                if self.weights_path is not None:
                    weight = sample['weight']
            output = self.model.forward(img)
            loss = self.losser.get_loss(output, label, weight, self.loss_name, self.labels_class)

            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        self.scheduler.step()

    def __val(self, epoch: int, dataloader, checkpoint_path: Path, logger, check_first_img=True) -> float:
        self.model.eval()
        val_metric = 0
        is_first = True
        with torch.no_grad():
            for sample in dataloader:
                if torch.cuda.is_available():
                    img, label = sample['img'].cuda(), sample['label'].cuda()
                    if self.weights_path is not None:
                        weight = sample['weight'].cuda()
                else:
                    img, label = sample['img'], sample['label']
                    if self.weights_path is not None:
                        weight = sample['weight']
                output_raw = self.model.forward(img)
                temp_metric = 0

                if self.val_metric_name == 'loss':
                    temp_metric = self.losser.get_loss(output_raw, label, weight, self.loss_name, self.labels_class).item()
                else:
                    output = output_raw.max(1)[1].data
                    output = output.cpu().squeeze().numpy()
                    label = sample['label'].squeeze().numpy().astype(np.int64)
                    temp_metric = metrics.get_metric(self.val_metric_name, output, label)

                val_metric += temp_metric
                # visualization
                if check_first_img and is_first and epoch % 1 == 0:
                    check_result(epoch, sample['img'][0, :, :, :].numpy().squeeze(),
                                 sample['label'].numpy().squeeze(),
                                 output_raw.max(1)[1].data.cpu().numpy().squeeze(),
                                 sample['weight'][0, :, :, :].squeeze().numpy().transpose((1, 2, 0)),
                                 checkpoint_path, logger)
                is_first = False
            val_metric /= len(dataloader.dataset)
            return val_metric

    def test(self, imgs_path: Path, labels_path: Path, test_files: List, pth_address: Path,
             test_vis_path: Path = None) -> float:
        try:
            net_inference = NetInference(self.gpu_detector, input_size=self.test_input_size,
                                         overlap_size=self.test_overlap_size,
                                         batch_size=self.test_batch_size, aug_type=self.test_aug_type,
                                         mean=self.imgs_mean, std=self.imgs_std,
                                         model=self.model_class, in_channel=self.input_channels,
                                         out_channel=self.labels_class,
                                         pth_address=pth_address, use_post_process=self.test_use_post_process)
        except AssertionError as e:
            print(e)
        except FileNotFoundError as e:
            print("The pth address is not right")

        test_metrics = 0.0
        for img_name in test_files:
            print("Analysing {} in test files".format(img_name))
            img = load_img(Path(imgs_path, img_name))
            label = load_img(Path(labels_path, img_name.replace(self.imgs_suffix, "npy")))
            output = net_inference.forward_one_image(img)
            temp_metric = metrics.get_metric(self.test_metric_name, output, label)
            test_metrics += temp_metric

            if test_vis_path is not None:
                test_vis_path.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(20, 20))
                plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("img"), plt.axis("off")
                plt.subplot(1, 3, 2), plt.imshow(label, cmap="gray"), plt.title("label"), plt.axis("off")
                plt.subplot(1, 3, 3), plt.imshow(output, cmap="gray"), plt.title("output"), plt.axis("off")
                plt.show()
                plt.savefig(str(Path(test_vis_path, img_name)))

        return test_metrics / len(test_files)


if __name__ == '__main__':
    seg_trainer = SegTrainer()

    cwd = os.getcwd()
    dataset_name = "isbi2012"
    data_bk_path = Path(cwd, "data", dataset_name, "data_backup")
    data_experiment_path = Path(cwd, "data", dataset_name, "data_experiment")
    parameters_path = Path(cwd, "experiment")

    use_preprocess = True

    seg_trainer.start_training(data_bk_path, data_experiment_path, parameters_path, use_preprocess)
