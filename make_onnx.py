import os
import io
import numpy as np

from torch import nn
# # import torch.utils.model_zoo as model_zoo
import torch.onnx
from network.net import net_option
import onnx

net = net_option(name = "Jacinto", mode = "segmentation")

# resume from checkpoint
checkpoint = "weights/end2end/Jacinto_end2end_bdd_bs_7_lr_0.0001_lambda_s_1.0_lambda_d_1.0_patiance_25_fix_backbone_False_fix_seg_False_fix_od_False_fix_bn_running_False_85.pth"
assert os.path.exists(checkpoint), "Checkpoint {} does not exist.".format(checkpoint)
state = torch.load(checkpoint)
net.load_state_dict(state["model_state"])
print("Resume from previous model {}".format(checkpoint))

# net = net.cuda()
net.target_available = False
# input
x = torch.randn(1, 3, 512, 1024, requires_grad=True)
# x = x.cuda()
# Export the model
torch_out = torch.onnx._export(net,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "test.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parame
