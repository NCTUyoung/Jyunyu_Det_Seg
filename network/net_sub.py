from __future__ import print_function, division
import os
# from model.mask import Mask
# from model.DRN import DRN_C_26
from network.JacintoNet import JacintoNet
# from model.Jacinto_sub import Jacinto_sub
from network.Jacinto_img import Jacinto_img
from network.Jacinto_one_channel import Jacinto_one_channel
from network.Jacinto_enhance_one_channel import Jacinto_enhance_one_channel_v1, Jacinto_enhance_one_channel_v2, Jacinto_enhance_one_channel_remove_sum
from network.Jacinto_pretrain import Jacinto_pretrain, Jacinto_pretrain_enhance
# from model.mask_sub import Mask_sub

def net_option(dataset, name = "JacintoNet", phase = "train"):
    if name == "DRN":
        net = DRN_C_26(num_classes = dataset.num_classes, input_size = dataset.size, phase = phase)
    elif name == "JacintoNet":
        net = JacintoNet(in_depth = 4, num_classes = 1, input_size = dataset.size, phase = phase)
        # net = JacintoNet(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase = phase)
    elif name == "Mask":
        net = Mask(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_sub":
        net = Jacinto_sub(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_one_channel":
        net = Jacinto_one_channel(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_enhance_one_channel_v1":
        net = Jacinto_enhance_one_channel_v1(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_enhance_one_channel_v2":
        net = Jacinto_enhance_one_channel_v2(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_enhance_one_channel_remove_sum":
        net = Jacinto_enhance_one_channel_remove_sum(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Mask_sub":
        net = Mask_sub(in_depth = dataset.in_depth, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_img":
        net = Jacinto_img(in_depth = 3, num_classes = 8, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_pretrain":
        net = Jacinto_pretrain(in_depth = 3, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    elif name == "Jacinto_pretrain_enhance":
        net = Jacinto_pretrain_enhance(in_depth = 3, num_classes = dataset.num_classes, input_size = dataset.size, phase= phase)
    else:
        raise RuntimeError("Model type doesn't define.")
    return net
    
if __name__ == "__main__":
    pass