from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils import *

# visualization
def vis_result(img: np.ndarray, label: np.ndarray, pred_x0: np.ndarray, output: np.ndarray, result_dir: Path,
               logger: Logger, weight: np.ndarray = None, epoch: int = None, description="val_", show_mask = False, isTrain = False) -> None:
    """
        Check visualization of result during training
        :param img:         the raw image      
        :param label:       the ground truth of image  
        :param output:      the output result drawn from network 
        :param result_dir:  the directory to storage visualization of output result
        :param logger:      the logger to record information in txt
        :param weight:      the weight map of label, None is not to visualization
        :param epoch:       the epoch index in the training
        :param description: the other description add before in the name of result
        :param show_mask:   wether to show label or output as mask in img

    Parameters
    ----------
    pred_x0
    """
    logger.log_print("Image size is {},    min is {}, max is {}".format(img.shape, np.amin(img), np.amax(img)))
    logger.log_print("Label size is {},    min is {}, max is {}".format(label.shape, np.amin(label), np.amax(label)))
    logger.log_print("Output size is {},   min is {}, max is {}".format(output.shape, np.amin(output), np.amax(output)))
    if weight is not None:
        logger.log_print("Weight-0 size is {}, min is {}, max is {}".format(weight[:, :, 0].shape, 
                                                                            np.amin(weight[:, :, 0]), np.amax(weight[:, :, 0])))
        logger.log_print("Weight-1 size is {}, min is {}, max is {}".format(weight[:, :, 1].shape, 
                                                                            np.amin(weight[:, :, 1]), np.amax(weight[:, :, 1])))
        
    
    plt.figure(figsize=(40, 40))
    column_num = 4
    if weight is not None:
        column_num = 6
    if isTrain:
        img = img.transpose(1, 2, 0)
    plt.subplot(1, column_num, 1), plt.imshow(img, cmap="gray"), plt.title("img"), plt.axis("off")
    if show_mask:
        label  = add_mask(label, img)
        output = add_mask(output, img)
        pred_x0 = add_mask(pred_x0, img)
        plt.subplot(1, column_num, 2), plt.imshow(label.astype(np.uint8)),  plt.title("label"),  plt.axis("off")
        plt.subplot(1, column_num, 3), plt.imshow(pred_x0.astype(np.uint8)),  plt.title("pred_x0"), plt.axis("off")
        plt.subplot(1, column_num, 4), plt.imshow(output.astype(np.uint8)), plt.title("output"), plt.axis("off")
    else:       
        plt.subplot(1, column_num, 2), plt.imshow(label,  cmap="gray"), plt.title("label"), plt.axis("off")
        plt.subplot(1, column_num, 3), plt.imshow(label.astype(np.uint8)), plt.title("pred_x0"), plt.axis("off")
        plt.subplot(1, column_num, 4), plt.imshow(output, cmap="gray"), plt.title("output"), plt.axis("off")
    
    if weight is not None:
        plt.subplot(1, column_num, 5), plt.imshow(weight[:, :, 0], cmap="plasma"), plt.title("weight-0"), plt.axis("off")
        plt.subplot(1, column_num, 6), plt.imshow(weight[:, :, 1], cmap="plasma"), plt.title("weight-1"), plt.axis("off")
    if epoch:
        plt.savefig(str(Path(result_dir, description + str(epoch).zfill(3) + "_result.png")))
    else:
        plt.savefig(str(Path(result_dir, description + "result.png")))
    plt.show()


def add_mask(label: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
        Add label with red mask on img, you can change your own visulization method in here
        :param label: label or output, the 1 value in it is reprsent by red color
        :param img:   the bottom image
        :return: mask image
    """
    result = np.expand_dims(label, 2).repeat(3, axis=2) * 255
    result[label == 1, :] = np.array([255, 0, 0])
    result = result * 0.4 + np.expand_dims(img, 2).repeat(3, axis=2) * 0.6
    return np.clip(result, 0, 255)


# plot training curve
def plot(epoch: int, train_value_list: List, val_value_list: List, checkpoint_path: Path, params: Dict, find_min_value: bool = True) -> None:
    """
        Plot and save training curve during training, and draw target epoch in the figure
        :param epoch: epoch index in the training
        :param train_value_list: the list of evaluation for training set, each element record metric after each epoch
        :param val_value_list:   the list of evaluation for val set, each element record metric after each epoch
        :param checkpoint_path:  the experiment directory to save the curve
        :param params: the dict variable which mantain the experiment parameters
        :param find_min_value:   True is to record max value, and False is to record min value by green line       
    """
    clear_output(True)
    curve_name   = params["val"]["metric_name"]
    ignore_epoch = params["val"]["ignore_show_epoch_num"]
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
    plt.plot(val_value_list,   color="b", label="val "   + curve_name)
    if len(val_value_list) > ignore_epoch:
        plt.axvline(x=val_value_list.index(target_value), ls='-', c='green')
        plt.legend(loc='best')
    plt.savefig(str(Path(checkpoint_path,  curve_name + '_curve.png')))
    plt.show()