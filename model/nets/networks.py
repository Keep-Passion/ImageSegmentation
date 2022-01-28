from .unet import UNet
from .resunet import ResUNet
from .attunet import AttUNet
from .nestedunet import NestedUNet


class Network:
    """
        Network class for network choosen, you can add your network in here
    """
    def __init__(self, net_name: str, use_pre_trained:bool = False):
        self.nets_name_list  = ["unet", "resunet", "attunet", "nestedunet"]
        self.nets_class_list = [UNet,    ResUNet,    AttUNet,  NestedUNet]
        self.net_name = net_name
        self.net_class = self.__check_nets()
      
    
    def __check_nets(self):
        """
        Check the self.net_name is wether supported in this framework
        """
        assert self.net_name in self.nets_name_list, "{} is not support in this code".format(self.net_name)
        return self.nets_class_list[self.nets_name_list.index(self.net_name)]
    
    
    def get_net(self, in_channels: int, out_channels: int, use_pre_trained: bool = False):
        # if self.net_name in ['', '']:
        #     return self.net_class(in_channels, out_channels, use_pre_trained = use_pre_trained)
        # else:
        #     return self.net_class(in_channels, out_channels)
        return self.net_class(in_channels, out_channels)
