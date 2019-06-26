import os
# import network.model_end2end as model
# import network.model_end2end_enhance_seg as model
# import network.model_end2end_enhance_seg_psp as model
# import network.model_end2end_enhance_seg_batch as model
# import network.model_end2end_enhance_seg_v2 as model
import network.model_end2end_enhance_seg_cat as model
# import network.model_end2end_share as model
# from network.Jacinto_detseg import Jacinto_detseg, Jacinto_detseg_OD_GUIDE_v1, Jacinto_detseg_OD_GUIDE_v2
# from network.Jacinto_detseg_double_anchor import Jacinto_detseg_double_anchor
from network.Jacinto_detseg_noup import Jacinto_detseg_noup
from network.Jacinto_detseg_v2 import Jacinto_detseg_v2
# from network.Jacinto_detseg_v3 import Jacinto_detseg_v3
# from network.Jacinto_detseg_v4 import Jacinto_detseg_v4
# from network.Jacinto_detseg_v5 import Jacinto_detseg_v5
# from network.Jacinto_detseg_v6 import Jacinto_detseg_v6
# from network.Jacinto_seg_experiment import Jacinto_seg_cat1, Jacinto_seg_cat2, Jacinto_seg_add1, Jacinto_seg_inter
from network.Jacinto_detseg_v2_binary import Jacinto_detseg_v2_binary
from network.Jacinto_det_256x512_64_1 import JacintoNet_det_256x512_64_1, JacintoNet_det_256x512_64_1_retinanet, JacintoNet_det_256x512_64_1_retinanet_only_encoder
from network.Jacinto_detseg_256x512 import Jacinto_detseg_256x512_v1, Jacinto_detseg_256x512_v4

def net_option(name = "Res", pretrain = None , phase = "train", depth = 50, input_shape = [512, 1024],mode = "segmentation", normalize_anchor = False, \
               num_classes_OD = 3, num_classes_seg = 8, use_focal_loss = False):
    print("num_classes_OD", num_classes_OD)
    print("num_classes_seg", num_classes_seg)
    print("use_focal_loss", use_focal_loss)
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
        # """ cityscape detection and segmentaion """
        # net = Jacinto_detseg(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 4, pretrained=pretrain, mode = mode)

        """ bdd100k detection 8cls and segmenation"""
        # net = Jacinto_detseg(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
        """ bdd100k detection 3cls and segmenation"""
        # net = Jacinto_detseg(num_classes_OD=3, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
        """ sur """
        net = Jacinto_detseg(num_classes_OD=3, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_double_anchor":
        # """ cityscape detection and segmentaion """
        net = Jacinto_detseg_double_anchor(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 4, pretrained=pretrain, mode = mode)

        """ bdd100k detection 8cls and segmenation"""
        # net = Jacinto_detseg(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
        """ bdd100k detection 3cls and segmenation"""
        # net = Jacinto_detseg(num_classes_OD=3, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_OD1":
        net = Jacinto_detseg_OD_GUIDE_v1(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_OD2":
        net = Jacinto_detseg_OD_GUIDE_v2(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_n":
        net = Jacinto_detseg_noup(num_classes_OD=8, num_classes_seg = 8, num_classes_lane = 4, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_v2":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v2(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 4,  pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v3":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v3(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 9, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v4":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v4(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v5":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v5(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_v6":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v6(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, input_shape = input_shape, mode = mode)
    elif name == "Jacinto_seg_cat1":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_seg_cat1(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_seg_cat2":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_seg_cat2(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_seg_add1":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_seg_add1(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_seg_inter":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_seg_inter(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_v2_binary":                                                               # input shape for OD anchor initial faster 
        net = Jacinto_detseg_v2_binary(num_classes_OD=8, num_classes_seg = 19, num_classes_lane = 5, pretrained=pretrain, mode = mode)
    elif name == "Jacinto_ssd_256x512":                                                               # input shape for OD anchor initial faster 
        net = JacintoNet_det_256x512_64_1(num_classes_OD = 4, normalize_anchor = normalize_anchor)
    elif name == "Jacinto_256x512_ret":                                                               # input shape for OD anchor initial faster 
        net = JacintoNet_det_256x512_64_1_retinanet()
    elif name == "Jacinto_256x512_ret_only":                                                               # input shape for OD anchor initial faster 
        net = JacintoNet_det_256x512_64_1_retinanet_only_encoder()
    elif name == "Jacinto_256x512_v1":
        net = Jacinto_detseg_256x512_v1(num_classes_OD=num_classes_OD, num_classes_seg = num_classes_seg, pretrained=pretrain, mode = mode, use_focal_loss = use_focal_loss)
    elif name == "Jacinto_256x512_v4":
        net = Jacinto_detseg_256x512_v4(num_classes_OD=num_classes_OD, num_classes_seg = num_classes_seg, pretrained=pretrain, mode = mode, use_focal_loss = use_focal_loss)
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