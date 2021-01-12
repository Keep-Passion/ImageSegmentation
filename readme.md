# The framework of image segmentation based on deep learning

This reprository is aimming to build a framework of image segmentation based on deep learning. You can easily add your own data, augmentation method, network, loss function, metric and tta method during network training.

For example, we provide a demo with neuro segementation task in [ISBI2012 challange](http://brainiac2.mit.edu/isbi_challenge/).

## Main points

class SegTrainer: Control the whole processing of training, validation and inference.

class DataAugmentor: Control the data augmentation method during training, you can easily add your own method in this class.

class Losser: Control the loss function during training, you can easily add your own method in this class.

class NetInference: Control the inference (test) process, including test time augmentation, overlap tile strategy and post processing

class OverlapTile: Control the overlaptile strategy during inference, please check the unet's paper

class TestTimeAug: Control the tta methods during inference, you can easily add your own method in this class.

Func get_metric: We provide 15 segmentation metrics in metrics.py and some details can be found at this [work](https://github.com/Keep-Passion/ImageSegMetrics).

Func data_preprocess: We define a format of data storage, you should change your own data according to required format.

Beside, we provide many designs in the code, such as cross validation, training log and many different implementations of unet weight map

## Requirement
All code are based on python36.  
Gala is installed according to https://github.com/janelia-flyem/gala.  