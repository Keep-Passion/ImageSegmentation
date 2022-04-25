import numpy as np
import cv2
from skimage import measure
import scipy.ndimage as ndimage
from skimage import morphology


class ErrorPenalizeDisWeight():
    """
    ErrorPenalizeDisWeight: 
    """
    def __init__(self, class_num=2):
        '''
        :param w0: 10 in the origin paper
        :param sigma:5 in the origin paper
        '''
        self._class_num = class_num
    
    
    def _get_weight(self, label):
        """
        返回根据Unet的方法计算的Weight图
        :param method: 选择什么样的方法，[0,1,2,3]
        :param neighbors: 求label时，4临接或8临接,默认4
        :param bgPixel: 背景像素，针对铝镧枝晶为0
        :return: 返回权重图weight
        """
        weight = np.zeros((label.shape[0], label.shape[1], self._class_num), dtype=np.float32)
        for class_idx in range(self._class_num):
            temp_label = np.zeros_like(label)
            temp_label[label == class_idx] = 1
            dis_trf = ndimage.distance_transform_edt(temp_label)
            max_value = np.amax(dis_trf)
            temp_weight = 1 + (max_value - dis_trf) / max_value
            weight[:, :, class_idx][label == class_idx] = temp_weight[label == class_idx]
        return weight
        