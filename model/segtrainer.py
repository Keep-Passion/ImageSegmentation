import os
from pathlib import Path
import h5py
from IPython.display import clear_output
import matplotlib.pyplot as plt

import cv2, time
import random
import skimage
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from sklearn.model_selection import KFold
import pandas as pd

from utils import *

import model.metrics as metrics
from .losses.losser import Losser
from .dataloader import SegDataset
from .seginferencer import SegInferencer
from .visualization import *
from .nets.networks import Network



class SegTrainer:
    """
        Segmentation class
    """
    def __init__(self, params: Dict, experiment_dir: Path, data_postprocess, ddpm, ddpm_opt, accelerator) -> None:
        
        self.params = params
        
        # model parameters
        self.net_class = Network(self.params["train"]["model_name"])
        self.model = None

        self.ddpm = ddpm
        self.ddpm_opt = ddpm_opt
        
        # data parameters
        self.imgs_path    = Path(Path(self.params["data"]["dir"]), "images")
        self.labels_path  = Path(Path(self.params["data"]["dir"]), "labels")
        self.weights_path = Path(Path(self.params["data"]["dir"]), self.params["data"]["weights_name"])
        self.all_files    = [str(item.name) for item in list(self.imgs_path.glob("*.{}".format(self.params["data"]["imgs_suffix"])))]
        self.train_files_list, self.val_files_list, self.test_files_list = self.__split_dataset()
        self.params["data"]["all_files"] = self.all_files
        self.params["data"]["train_files_list"] = self.train_files_list
        self.params["data"]["val_files_list"]   = self.val_files_list
        self.params["data"]["test_files_list"]  = self.test_files_list
        
        
        # training parameters
        setup_seed(self.params["train"]["seed"])
        self.is_min_val_metirc = self.__check_min_val_metric(self.params["val"]["metric_name"])
        self.experiment_dir = experiment_dir
        ParamsYml().save_params(Path(self.experiment_dir, "experiment_params.yml"), params = self.params)
        
        # loss parameters
        self.losser = Losser()

        # ddpm
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.accelerator = accelerator
         
        # gpu parameters
        self.gpu_detector = GpuDetector()
        try:
            self.gpu_detector.set_gpu_device(self.params["gpu"]["cuda_devices"])
        except AssertionError as e:
            print(e)      
        
        # test parameters
        self.seginferencer = SegInferencer(self.ddpm, self.experiment_dir.name, self.accelerator, is_test_all=False, data_postprocess=data_postprocess)
        
        self.logger = Logger(True, Path(self.experiment_dir, "train_log.txt"))
        
    
    def __check_min_val_metric(self, val_metric_name) -> bool:
        if val_metric_name in ["loss", "vi"]:
            return True
        else:
            return False
        
    
    def __split_dataset(self) -> Tuple:
        """
            Split dataset according to cross validation
        """
        train_files_list, val_files_list, test_files_list =[], [], []
        kf = KFold(n_splits = self.params["data"]["kf_num"], shuffle = True, random_state = self.params["train"]["seed"])
        for kf_idx, (train_files, test_files) in enumerate(kf.split(self.all_files)):
            val_files = random.sample(list(train_files), int(len(train_files) * self.params["data"]["val_rate"]) + 1 )
            train_files = list(set(train_files).difference(set(val_files)))
            train_files.sort(); val_files.sort() 
            
            train_files = [self.all_files[item] for item in train_files]
            val_files   = [self.all_files[item] for item in val_files]
            test_files  = [self.all_files[item] for item in test_files]
            train_files_list.append(train_files)
            val_files_list.append(val_files)
            test_files_list.append(test_files)
        
        return train_files_list, val_files_list, test_files_list
    
    
    def start_training(self) -> None:
        test_metric_list = []
           
        self.logger.log_print("Experiment: {}".format(str(self.experiment_dir.name)))
        
        # K-fold Cross Validation
        for kf_idx in range(self.params["data"]["kf_num"]):
          
            # train setting
            if self.params["data"]["use_kfold"]:
                # mkdir each folder for parameters dir
                temp_parameters_dir = Path(self.experiment_dir, str(self.params["data"]["kf_num"]) + '_fold_' + str(kf_idx))
                temp_parameters_dir.mkdir(parents=True, exist_ok=True)
                self.logger.log_print("Cross Validation: {}".format(str(self.params["data"]["kf_num"])  + '_fold_' + str(kf_idx)))
            else:
                temp_parameters_dir = self.experiment_dir
                if kf_idx >= 1:
                    break
            train_files = self.train_files_list[kf_idx]
            val_files   = self.val_files_list[kf_idx]
            test_files  = self.test_files_list[kf_idx]
            
            train_dataset = SegDataset(self.params, train_files, use_augment = True)
            val_dataset   = SegDataset(self.params, val_files, use_augment = False)
            
            train_loader     = DataLoader(train_dataset, batch_size = self.params["train"]["batch_size"], 
                                          shuffle = True,  pin_memory=self.params["train"]["pin_memory"], num_workers = self.params["train"]["num_worker"])
            train_val_loader = DataLoader(train_dataset, batch_size = self.params["val"]["batch_size"],   
                                          shuffle = False, pin_memory=self.params["train"]["pin_memory"], num_workers = self.params["train"]["num_worker"])
            val_loader       = DataLoader(val_dataset,   batch_size = self.params["val"]["batch_size"],   
                                          shuffle = False, pin_memory=self.params["train"]["pin_memory"], num_workers = self.params["train"]["num_worker"])
            
            self.model = self.net_class.get_net(self.params["data"]["imgs_channels"], self.params["data"]["labels_class_num"])
            
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.params["train"]["lr"])
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)
            
            if torch.cuda.is_available() and self.params["gpu"]["use_gpu"]:
                self.model = nn.DataParallel(self.model, device_ids=self.gpu_detector.get_current_gpu()[1]).cuda()

            train_metric_list = []
            val_metric_list = []
            
            val_baseline = 0
            if self.is_min_val_metirc:
                val_baseline = 10000

            val_best_epoch = 0
            val_metric_name = self.params["val"]["metric_name"]

            # if self.accelerator.num_processes == 1 and torch.cuda.device_count() > 1:
            #     self.model = nn.DataParallel(self.model)
                # self.ddpm = nn.DataParallel(self.ddpm)

            
            # Training
            st_total = time.time()
            self.logger.log_print("Training:")

            # device
            self.model, self.ddpm, self.optimizer, self.ddpm_opt, train_loader, self.scheduler = self.accelerator.prepare(self.model, self.ddpm, self.optimizer,
                                                                                            self.ddpm_opt, train_loader,
                                                                                            self.scheduler)

            #------------------------------- epoch begin-------------------------------------------#
            for i in range(1, self.params["train"]["epochs"] + 1):
                self.logger.log_print("Experiment name: " + str(self.experiment_dir.name))
                st = time.time()
                
                st_train = time.time()

                # every epoch train
                self.__train(train_loader, i)
                ed_train = time.time()
                self.logger.log_print("Train Duration: {}'s".format(ed_train - st_train))

                # every manual valid
                st_val = time.time()
                train_metric = self.__val(i, train_val_loader, temp_parameters_dir, self.logger,  vis_first_img=False)
                val_metric   = self.__val(i, val_loader, temp_parameters_dir, self.logger, vis_first_img=self.params["val"]["is_show_first_img"])
                ed_val = time.time()
                self.logger.log_print("Val Duration {}'s".format(ed_val - st_train))
                
                train_metric_list.append(train_metric)
                val_metric_list.append(val_metric)
                
                self.logger.log_print("Epoch {}: train_{} {:.4f}; val_{} {:.4f} \n".format(i, val_metric_name, train_metric_list[-1], val_metric_name, val_metric_list[-1]))
                
                plot(i, train_metric_list, val_metric_list, temp_parameters_dir, self.params, self.is_min_val_metirc)

                if (i > self.params["val"]["ignore_show_epoch_num"]) and ((self.is_min_val_metirc and val_metric < val_baseline) or (self.is_min_val_metirc is False and val_metric > val_baseline)): 
                    val_baseline = val_metric
                    val_best_epoch = i
                    torch.save(self.model.state_dict(), str(Path(temp_parameters_dir, "best_model_state.pth")))

                    ddpm_ckpt = {
                        'step': 0,
                        'model': self.accelerator.get_state_dict(self.ddpm),
                        'opt': self.ddpm_opt.state_dict(),
                        'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
                        'version': '1.9.4'
                    }

                    torch.save(ddpm_ckpt, str(Path(temp_parameters_dir, "best_ddpm_model_state.pt")))
                    del ddpm_ckpt

                ed = time.time()
                self.logger.log_print("Epoch Duration: {}'s".format(ed - st))
            ed_total = time.time()
            self.logger.log_print("Total duration is: {}'s".format(ed_total - st_total))
            self.logger.log_print("The best epoch is at: {} th epoch".format(val_best_epoch))
            self.logger.log_print("Train {} list is: {}".format(val_metric_name, train_metric_list))                        
            self.logger.log_print("Val   {} list is: {}".format(val_metric_name, val_metric_list))
            
            self.best_model_path = Path(temp_parameters_dir, "best_model_state.pth")
            self.best_ddpm_model_state = Path(temp_parameters_dir, "best_ddpm_model_state.pth")

            torch.cuda.empty_cache()

            test_metric_list.append(self.seginferencer.test(self.best_model_path, self.best_ddpm_model_state, test_files, 'test_vis'))
            self.logger.log_print("test_metrics is {}".format(test_metric_list))
        self.logger.log_print("Experiment is {}".format(str(self.experiment_dir)))
        self.logger.log_print("Metric is {} in test time".format(self.params["test"]["metric_name"]))
        if self.params["data"]["use_kfold"]:
            self.logger.log_print("For cross validation, mean is: {}, std is {}".format(np.mean(test_metric_list), np.std(test_metric_list)))
        else:
            self.logger.log_print("The test value is: {}".format(test_metric_list[0]))


    def __train(self, dataloader, epoch):
        self.model.train()
        self.ddpm.train()

        #----------------- batch ------------------------#
        for sample in dataloader: # 2D [B, C, H, W]
            imgs   = load_tensor_to_cgpu(sample['img'], self.params, self.gpu_detector.get_current_gpu()[1])
            labels = load_tensor_to_cgpu(sample['label'], self.params, self.gpu_detector.get_current_gpu()[1])
            weights = None
            if self.params["data"]["weights_name"] is not None:
                weights = load_tensor_to_cgpu(sample['weight'], self.params, self.gpu_detector.get_current_gpu()[1])

            #-------------------------- ddpm training ------------------------#
            #-------------------------- batch --------------------------------#

            with self.accelerator.autocast():
                imgs = imgs.to(self.accelerator.device)
                labels = labels.to(self.accelerator.device)
                # concat img & label
                gen_input = torch.concat((imgs, labels), dim=0)
                loss_diff, img_pred = self.ddpm(gen_input)
                loss_diff = loss_diff.mean()
                print("epoch: {} , loss_gen:{}".format(epoch, loss_diff))
                # img_pred = self.upconv(img_pred)
                img_pred = img_pred.to(self.accelerator.device)
                outputs = self.model.forward(img_pred)
                loss = self.losser.get_loss(outputs, labels, weights, self.params["train"]["loss_name"])
                print("epoch: {} , loss_seg:{}".format(epoch, loss))
                loss_total = loss + loss_diff
                loss_items2 = torch.tensor([loss_diff]).to(self.accelerator.device).detach()
                loss_items = loss.detach().item()
                # loss_items = torch.cat((loss_items, loss_items2))

                if self.accelerator.num_processes != 1:
                    loss_total *= self.accelerator.num_processes  # gradient averaged between devices in DDP mode
                # if opt.quad:
                #     loss_total *= 4.
            # Backward
            self.accelerator.backward(loss_total)

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            # Optimize
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.ddpm_opt.step()
            self.ddpm_opt.zero_grad()
            # if self.params["data"]["ema"]:
            #     ema.update(model)
            #     ddpm_ema.update()

        self.scheduler.step()
    
    def __val(self, epoch: int, dataloader, checkpoint_path: Path, logger, vis_first_img=True) -> float:
        self.model.eval()
        self.ddpm.eval()
        val_metric = 0
        is_first = True
        with torch.no_grad():
            for sample in dataloader: 
                # inference
                imgs   = load_tensor_to_cgpu(sample['img'], self.params, self.gpu_detector.get_current_gpu()[1])
                labels = load_tensor_to_cgpu(sample['label'], self.params, self.gpu_detector.get_current_gpu()[1])
                if self.params["data"]["weights_name"] is not None:
                    weights = load_tensor_to_cgpu(sample['weight'], self.params, self.gpu_detector.get_current_gpu()[1])  

                # ddpm inference
                # for i in range(imgs.size(0)):
                #     img = imgs[i]
                #     label = labels[i]
                #     weight = None
                #     if weights is not None:
                #         weight = weights[i]

                with self.accelerator.autocast():
                    imgs = imgs.to(self.accelerator.device)
                    labels = labels.to(self.accelerator.device)
                    loss_diff, img_pred = self.ddpm(imgs, labels)
                    outputs = self.model.forward(img_pred)

                if self.params["val"]["metric_name"] == 'loss':
                    val_metric += self.losser.get_loss(outputs, labels, weights,
                                                       self.params["train"]["loss_name"]).item()
                else:
                    for idx in range(sample['img'].size()[0]):
                        output = outputs[idx:idx+1, :, :, :].max(1)[1].data.cpu().squeeze().numpy()
                        label = labels[idx:idx+1, :, :, :].squeeze().cpu().numpy().astype(np.int64)
                        val_metric += metrics.get_metric(self.params["val"]["metric_name"], output, label)

                # outputs = self.model.forward(imgs)

                # # evaluation
                # if self.params["val"]["metric_name"] == 'loss':
                #     val_metric += self.losser.get_loss(outputs, labels,weights, self.params["train"]["loss_name"]).item()
                # else:
                #     for idx in range(sample['img'].size()[0]):
                #         output = outputs[idx:idx+1, :, :, :].max(1)[1].data.cpu().squeeze().numpy()
                #         label  = labels[idx:idx+1, :, :, :].squeeze().cpu().numpy().astype(np.int64)
                #         val_metric += metrics.get_metric(self.params["val"]["metric_name"], output, label)

                # visualization
                if vis_first_img and is_first and epoch % self.params["val"]["show_interval"] == 0:
                    vis_result(sample['img'][0, :, :, :].numpy().squeeze(),
                               sample['label'][0, :, :, ].numpy().squeeze(),
                               img_pred[0:1, :, :, :].max(1)[1].data.cpu().squeeze().numpy(),
                               outputs[0:1, :, :, :].max(1)[1].data.cpu().squeeze().numpy(), 
                               checkpoint_path, logger,
                               weight = sample['weight'][0, :, :, :].squeeze().numpy().transpose((1,2,0)),
                               epoch = epoch,
                               description="val_", show_mask = False, isTrain=True)
                is_first = False

            return val_metric / len(dataloader)