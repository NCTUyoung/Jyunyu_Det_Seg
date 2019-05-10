#####################################
# Define parameters of each network
#####################################

class model_end2end_enhance_seg_cat(object):
    # model parameters
    resnet_backbone_parameters = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4"]
    fpn_segmentation_parameters = ["K5_conv_1x1", "K4_conv_1x1", "K3_conv_1x1","K2_conv_1x1","K5_top", "K4_top","K3_top","K2_top"]
    segmentation_subnet_parameters = ["D2_conv_3x3_1","D2_conv_3x3_2","D3_conv_3x3_1","D3_conv_3x3_2","D4_conv_3x3_1","D4_conv_3x3_2", \
                                  "D5_conv_3x3_1","D5_conv_3x3_2", "upsample1","upsample2","upsample3","conv_final_smooth",
                                  "conv_up1", "conv_up2", "conv_final", "seg_loss"]
    fpn_detection_parameters = ["P5_conv","P5_top","P4_conv","P4_top","P3_conv","P3_top","P6_conv","P7_conv"] 
    detection_subnet_parameters = ["regressionModel","classificationModel"]   
    @classmethod
    def fix_parameters(cls, net, logger, *args):
        # if fix_backbone:
        if args[0]:
            count = 0
            for name, module in net.named_children():
                if name in cls.resnet_backbone_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed resnet backbone: {}/{}".format(count,len(cls.resnet_backbone_parameters)))

        # if fix_segmentation:
        if args[1]:
            count = 0
            for name, module in net.fpn.named_children():
                if name in cls.fpn_segmentation_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed fpn segmentation: {}/{}".format(count,len(cls.fpn_segmentation_parameters)))

            count = 0
            for name, module in net.named_children():
                if name in cls.segmentation_subnet_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed segmentation subnet: {}/{}".format(count,len(cls.segmentation_subnet_parameters)))

        # if fix_detdection:
        if args[2]:
            count = 0
            for name, module in net.fpn.named_children():
                if name in cls.fpn_detection_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed fpn detection: {}/{}".format(count,len(cls.fpn_detection_parameters)))

            count = 0
            for name, module in net.named_children():
                if name in cls.detection_subnet_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed detection subnet: {}/{}".format(count,len(cls.detection_subnet_parameters)))

class Jacinto_det_seg(object):
    Jacinto_backbone_parameters = ["conv1a", "conv1b", "res2a_branch2a", "res2a_branch2b", "res3a_branch2a", "res3a_branch2b", "res4a_branch2a", \
                            "res4a_branch2b", "res5a_branch2a", "res5a_branch2b"]
    Jacinto_segmentation_parameters = ["out3a", "out5a", "ctx_conv1", "ctx_conv2", "ctx_conv3", "ctx_conv4", "ctx_conv_up1", \
                            "ctx_conv_up2", "ctx_conv_up3", "ctx_final", "seg_loss"]
    Jacinto_detection_parameters = ["ctx_conv1_det", "ctx_conv2_det", "ctx_conv3_det", "ctx_conv4_det", "ctx_conv5_det", \
                            "regressionModel", "classificationModel"]
    Jacinto_lane_parameters = ["ctx_conv_up1_lane", "ctx_conv_up2_lane", "ctx_conv_up3_lane", "ctx_conv_up3_class", "ctx_final_lane", \
                            "binary_class_loss", "lane_loss"]
    @classmethod
    def fix_parameters(cls, net, logger, *args):
        # fix backbone
        if args[0]:
            count = 0
            for name, module in net.named_children():
                if name in cls.Jacinto_backbone_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed Jacinto backbone: {}/{}".format(count,len(cls.Jacinto_backbone_parameters)))
        # fix segmentation
        if args[1]:
            count = 0
            for name, module in net.named_children():
                if name in cls.Jacinto_segmentation_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed Jacinto segmentation: {}/{}".format(count, len(cls.Jacinto_segmentation_parameters)))
        # fix detection
        if args[2]:
            count = 0
            for name, module in net.named_children():
                if name in cls.Jacinto_detection_parameters:
                    count += 1
                    for para in module.parameters():
                        para.requires_grad = False
            logger.info("Fixed detection subnet: {}/{}".format(count,len(cls.Jacinto_detection_parameters)))

def fix_model_parameters(net, type, logger, *args):
    # if type == "model_end2end_enhance_seg_cat":
    if type == "Res":
        model_end2end_enhance_seg_cat.fix_parameters(net, logger, *args)
    elif type == "Jacinto" or type == "Jacinto_v2" or type == "Jacinto_v3" or type == "Jacinto_v4" or type == "Jacinto_v5"\
     or type == "Jacinto_OD1" or type == "Jacinto_OD2":
        Jacinto_det_seg.fix_parameters(net, logger, *args)
    else:
        logger("Model type does not define.")
        raise 

