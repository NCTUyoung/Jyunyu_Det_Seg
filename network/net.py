import os
# import network.model_end2end as model
# import network.model_end2end_enhance_seg as model
# import network.model_end2end_enhance_seg_psp as model
# import network.model_end2end_enhance_seg_batch as model
# import network.model_end2end_enhance_seg_v2 as model
import network.model_end2end_enhance_seg_cat as model
# import network.model_end2end_share as model
from network.Jacinto_detseg import Jacinto_detseg, Jacinto_detseg_OD_GUIDE_v1, Jacinto_detseg_OD_GUIDE_v2
from network.Jacinto_detseg_noup import Jacinto_detseg_noup
from network.Jacinto_detseg_v2 import Jacinto_detseg_v2
from network.Jacinto_detseg_lane_v3 import Jacinto_detseg_lane_v3
from network.Jacinto_detseg_lane_v4 import Jacinto_detseg_lane_v4
from network.Jacinto_detseg_lane_v5 import Jacinto_detseg_lane_v5
from network.Jacinto_detseg_v2_binary import Jacinto_detseg_v2_binary


def net_option(name = "Res", pretrain = None , phase = "train", depth = 50, input_shape = [512, 1024],mode = "segmentation"):
    if name == "Res" and depth == 18:
        net = model.resnet18(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = mode)#End2End
    elif name == "Res" and depth == 34:
        net = model.resnet34(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = mode)
    elif name == "Res" and depth == 50:
        net = model.resnet50(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = mode)
    elif name == "Res" and depth == 152:
        net = model.resnet101(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = mode)
    elif name == "Res" and depth == 101:
        net = model.resnet152(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = mode)
    elif name == "Jacinto":
        net = Jacinto_detseg(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_OD1":
        net = Jacinto_detseg_OD_GUIDE_v1(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_OD2":
        net = Jacinto_detseg_OD_GUIDE_v2(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_n":
        net = Jacinto_detseg_noup(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_v2":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v2(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 4,  pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v3":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_lane_v3(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 9, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v4":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_lane_v4(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v5":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_lane_v5(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v2_binary":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v2_binary(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, input_shape = input_shape, mode = mode)
    else:
        raise RuntimeError("Model type doesn't define.")
    return net

# def segmentation_net(dataset, pretrain = False ,name = "Res", phase = "train", depth = 50):
#     if name == "Res" and depth == 18:
#         net = model.resnet18(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = "segmentation")
#     elif name == "Res" and depth == 34:
#         net = model.resnet34(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = "segmentation")
#     elif name == "Res" and depth == 50:
#         net = model.resnet50(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = "segmentation")
#     elif name == "Res" and depth == 101:
#         net = model.resnet152(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = "segmentation")
#     elif name == "Res" and depth == 152:
#         net = model.resnet152(num_classes_OD=8, num_classes_seg = 19, pretrained=pretrain, mode = "segmentation")
#     elif name == "JacintoNet":
#         net = JacintoNet(num_classes = dataset.num_classes, input_size = dataset.size, phase = phase)
#     else:
#         raise RuntimeError("Model type doesn't define.")
#     return net
    
if __name__ == "__main__":
    pass