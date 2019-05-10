#####################################################################
# Jacinto based model detection segmentation and lane detection
# More layer for lane detection to see if it can get higher accuracy
# pretrain using cityscape semantic segmentation and detection to 
# train lane detection on cuLane
# Date: 2019/03/26
#####################################################################
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.net_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from network.anchors import Anchors
from network.loss_seg import CrossEntropyLoss2d 
import network.losses as losses
print(torch.__version__)
if torch.__version__[0] == "0":
    from lib.nms.pth_nms import pth_nms as nms_lib
    def nms(dets, thresh):
        "Dispatch to either CPU or GPU NMS implementations.\
        <Acc></Acc>ept dets as tensor"""
        return nms_lib(dets, thresh)
elif torch.__version__[0] == "1":
    from network.nms import boxes_nms as nms
# from utils.core_utils import load_model

# pretrain_model = os.path.join("weights", \
# "Jacinto_pretrain_enhance_256x256_bs_5_lrsh_True_pretrain_True_ad_box_True__ignore_balance_data_loss_selection_no255_one_channel_io_RGB_img_one_channel_360.pth)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
                                                
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)
        # print("out.size()",out.size())
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class Jacinto_detseg_lane_v3(nn.Module):
    def __init__(self, num_classes_seg, num_classes_OD, num_classes_lane, pretrained=None, input_shape = [512,1024], mode = "detection"):
        super(Jacinto_detseg_lane_v3, self).__init__()
        self.num_classes_seg = num_classes_seg
        self.num_classes_OD = num_classes_OD
        self.num_classes_lane = num_classes_lane
        self.pretrained = pretrained
        self.mode = mode
        self.target_available = True
        self.nms_threshold = 0.4
        self.input_shape = input_shape

        ###################
        # Batchnorm layer #
        ###################
        # batchnorm layer to fix during training lane network
        self.bn_in_layers = ["conv1a", "conv1b", "res2a_branch2a", "res2a_branch2b", "res3a_branch2a", "res3a_branch2b", "res4a_branch2a", "res4a_branch2b",\
                        "res5a_branch2a", "res5a_branch2b", "out3a", "out5a", "ctx_conv1", "ctx_conv2", "ctx_conv3", "ctx_conv4",\
                        "ctx_conv_up1", "ctx_conv_up2", "ctx_conv_up3"]

        self.conv1a = self._make_conv_block(3, 32, 5, 2, 2, 1, 1)
        self.conv1b = self._make_conv_block(32, 32, 3, 1, 1, 1, 4) 
        
        self.res2a_branch2a = self._make_conv_block(32, 64, 3, 1, 1, 1, 1) 
        self.res2a_branch2b = self._make_conv_block(64, 64, 3, 1, 1, 1, 4) 
        
        self.res3a_branch2a = self._make_conv_block(64, 128, 3, 1, 1, 1, 1) 
        self.res3a_branch2b = self._make_conv_block(128, 128, 3, 1, 1, 1, 4)
        
        self.res4a_branch2a = self._make_conv_block(128, 256, 3, 1, 1, 1, 1) 
        self.res4a_branch2b = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        
        self.res5a_branch2a = self._make_conv_block(256, 512, 3, 1, 2, 2, 1) 
        self.res5a_branch2b = self._make_conv_block(512, 512, 3, 1, 2, 2, 4)

        self.out3a = self._make_conv_block(128, 64, 3, 1, 1, 1, 2) 
        self.out5a = self._make_conv_block(512, 64, 3, 1, 4, 4, 2) 
        
        self.ctx_conv1 = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv2 = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv3 = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv4 = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        ###########
        # enhance #
        ###########
        self.ctx_conv_up1 = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up2 = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up3 = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_final = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1),
                            nn.ReLU())
        self.seg_loss = CrossEntropyLoss2d()
        #############
        # detection #
        #############
        self.ctx_conv1_det = nn.Conv2d(64, 64, 3, 1, 1, 1 ,1) #128x64
        self.ctx_conv2_det = nn.Conv2d(64, 64, 3, 1, 1, 1 ,1) # 64x32
        self.ctx_conv3_det = nn.Conv2d(64, 64, 3, 1, 1, 1 ,1) # 32x16
        self.ctx_conv4_det = nn.Conv2d(64, 64, 3, 2, 1, 1 ,1) # 16x8
        self.ctx_conv5_det = nn.Conv2d(64, 64, 3, 2, 1, 1 ,1) # 8x4

        self.regressionModel = RegressionModel(64+8, num_anchors = 15)
        self.classificationModel = ClassificationModel(64+8, num_anchors = 15, num_classes= self.num_classes_OD)

        self.anchors = Anchors(input_shape = self.input_shape)
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()   
        self.focalLoss = losses.FocalLoss()

        ########
        # Lane #
        ########
        self.ctx_conv_up1_1_lane = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv_up1_2_lane = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv_up1_3_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up1_4_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up1_5_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up2_1_lane = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv_up2_2_lane = self._make_conv_block(64, 64, 3, 1, 4, 4, 1)
        self.ctx_conv_up2_3_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up2_4_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up2_5_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up3_lane = self._make_conv_block(64, 64, 3, 1, 1, 1, 1)
        self.ctx_conv_up3_class = self._make_conv_block(64, self.num_classes_lane, 3, 1, 1, 1, 1)
        # self.ctx_conv_up3_class = self._make_conv_block(self.num_classes_lane-1, self.num_classes_lane-1, 3, 1, 1, 1, 1)
        # self.ctx_conv_up3_class = nn.Conv2d(64, self.num_classes_lane, 3, 1, 1, 1 ,1)

        self.ctx_final_lane = nn.Sequential(nn.Conv2d(64, self.num_classes_lane, 3, 1, 1, 1 ,1),
                            nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.binary_class_loss = torch.nn.BCELoss()
        self.lane_loss = CrossEntropyLoss2d(weight = True)
        self.softmax = torch.nn.Softmax(dim = 3)
        #############################
        # up and downsample formula #
        #############################
        self.poolind2d = nn.MaxPool2d(2,2)
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size = (512,1024))
        # if self.pretrained:
        #     assert os.path.exists(self.pretrained),"Pretrain model {} does not exist.".format(self.pretrained)
        #     loaded_state = torch.load(self.pretrained)
        #     net.load_state_dict(loaded_state["model_state"], strict=False)
        #     print("Resume from pretrain model {}".format(self.pretrained))

    def upsample_back(self, target, pred):
        """This function upsample pred to target's size
        Args:
        (1) target: segmentation target [b, H, W]
        (2) pred  : network prediction 
        Return 
        (2) pred  : prediction with same size with target
        """
        # print("target.size(), pred.size()", target.size(), pred.size())
        if (target.size()[1], target.size()[2]) != (pred.size()[2], pred.size()[3]):
            pred = F.interpolate(pred, (target.size()[1], target.size()[2]), mode='nearest')
            # print(pred.size())
            return pred
        else :
            return pred

    def _make_conv_block(self, in_channel, out_channel, kernel_size, stride, padding, dilation = 1, group = 1):
        layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation ,group),
                               nn.BatchNorm2d(out_channel),
                            nn.ReLU()
        )
        return layers

    # def Bilinear_Interpolation(self, x):
    #     print(x.size())
    #     x = F.interpolate(input, (int(x.size()[2]*2), int(x.size()[3]*2)), mode = 'bilinear')
    #     return x
    def freeze_bn(self):
        '''Freeze BatchNorm running statictic.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_bn_layers(self):
        '''Freeze BatchNorm running statictic.'''
        for name, module in self.named_children():
            if name in self.bn_in_layers:
                for layer in module:
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.eval()

    def forward(self, inputs):
        if self.mode == "segmentation":
            return self.forward_segmentation(inputs)
        elif self.mode == "detection":
            return self.forward_detection(inputs)
        elif self.mode == "lane":
            return self.forward_lane(inputs)
        else:
            if self.target_available:
                img_batch, annot_det, annot_seg = inputs
            else:
                img_batch = inputs     # 1024x512
            x = self.conv1a(img_batch) # 512x256
            x = self.conv1b(x)   
            x = self.poolind2d(x)
            x = self.res2a_branch2a(x) # 256x128
            x = self.res2a_branch2b(x)
            x = self.poolind2d(x)
            x = self.res3a_branch2a(x) # 128x64
            x = self.res3a_branch2b(x)
            out3a_ = self.out3a(x)
            x = self.poolind2d(x)
            x = self.res4a_branch2a(x) # 64x32
            x = self.res4a_branch2b(x)
            # x = self.poolind2d(x)    # only detection do this

            x = self.res5a_branch2a(x) 
            x = self.res5a_branch2b(x)
            x = self.out5a(x)
            out5a_up2 = self.bilinear2d(x)
            out5a_combined = out5a_up2 + out3a_ # 128 x 64

            x = self.ctx_conv1(out5a_combined)
            x = self.ctx_conv2(x)
            x = self.ctx_conv3(x)
            x = self.ctx_conv4(x)                                      #at 128x64 , 64x32 , 32x16 , 16x8 , 8x4
            ###################
            # enhance decoder #
            ###################
            x = self.bilinear2d(x) 
            # x = torch.cat((x, out2),1)
            x = self.ctx_conv_up1(x) # 256x128
            x = self.bilinear2d(x) 
            x = self.ctx_conv_up2(x) # 512x256
            x = self.bilinear2d(x) # 
            x = self.ctx_conv_up3(x) # 1024x512
            pred_seg = self.ctx_final(x)

            
            #############
            # detection #
            #############
            ctx_out1      = self.ctx_conv1_det(out5a_combined) #128x64
            ctx_out1_down = self.poolind2d(ctx_out1)
            ctx_out2      = self.ctx_conv2_det(ctx_out1_down)  #64x32
            ctx_out2_down = self.poolind2d(ctx_out2)
            ctx_out3      = self.ctx_conv3_det(ctx_out2_down) #32x16
            # ctx_out3_down = self.poolind2d(ctx_out3)
            ctx_out4      = self.ctx_conv4_det(ctx_out3) #16x8
            # ctx_out4_down = self.poolind2d(ctx_out4)
            ctx_out5      = self.ctx_conv5_det(ctx_out4) #8x4

            ############################################
            # segmentation output connect to detection #
            ############################################
            pred_seg_down = self.maxpool(pred_seg[:,11:,:,:]) # 512X256
            pred_seg_down = self.maxpool(pred_seg_down) # 256X128
            pred_seg_down = self.maxpool(pred_seg_down) # 128X64
            ctx_out1 = torch.cat([ctx_out1, pred_seg_down], dim = 1)
            pred_seg_down = self.maxpool(pred_seg_down) # 64X32
            ctx_out2 = torch.cat([ctx_out2, pred_seg_down], dim = 1)
            pred_seg_down = self.maxpool(pred_seg_down) # 32X16
            ctx_out3 = torch.cat([ctx_out3, pred_seg_down], dim = 1)

            pred_seg_down = self.maxpool(pred_seg_down) # 16x8
            ctx_out4 = torch.cat([ctx_out4, pred_seg_down], dim = 1)
            pred_seg_down = self.maxpool(pred_seg_down) # 16x8
            ctx_out5 = torch.cat([ctx_out5, pred_seg_down], dim = 1)     

            regression = torch.cat([self.regressionModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
            anchors = self.anchors(img_batch)

            if self.training and self.target_available:
                focal_loss = self.focalLoss(classification, regression, anchors, annot_det)
                pred_seg = self.upsample_back(annot_seg, pred_seg)
                seg_loss = self.seg_loss(pred_seg, annot_seg)
                return [focal_loss, seg_loss, pred_seg]
            else:
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
                # print(classification)
                # print(classification.size())
                scores = torch.max(classification, dim=2, keepdim=True)[0]
                # print(scores)
                # print(scores.size())
                scores_over_thresh = (scores>0.05)[0, :, 0]
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just return
                    return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), pred_seg]

                classification = classification[:, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                scores = scores[:, scores_over_thresh, :]
                tic = time.time()
                if torch.__version__[0] == "0":
                    anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], self.nms_threshold)
                else:
                    transformed_anchors = torch.squeeze(transformed_anchors)
                    scores = torch.squeeze(scores)
                    anchors_nms_idx = nms(transformed_anchors, scores, self.nms_threshold)
                toc = time.time()
                # print("NMS took {}".format(toc-tic))
                transformed_anchors = transformed_anchors.squeeze()
                nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
                # segmentation
                pred_seg = pred_seg #8x downsample
                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
                return [nms_scores, nms_class, transformed_anchors[anchors_nms_idx, :], pred_seg]

    def forward_segmentation(self, inputs):
        if self.target_available:
            img_batch, target = inputs
        else:
            img_batch = inputs
        x = self.conv1a(img_batch) # 1024x 512
        x = self.conv1b(x) 
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) # 512x256
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) # 256x128
        x = self.res3a_branch2b(x)
        out3a_ = self.out3a(x)
        x = self.poolind2d(x)
        x = self.res4a_branch2a(x) # 128x64
        x = self.res4a_branch2b(x)
        # x = self.poolind2d(x)

        x = self.res5a_branch2a(x) # 64x32
        x = self.res5a_branch2b(x)
        x = self.out5a(x)
        # out5a_up2 = self.Bilinear_Interpolation(x)
        out5a_up2 = self.bilinear2d(x)
        # print(out3a_.size())
        # print(out5a_up2.size())
        out5a_combined = out5a_up2 + out3a_

        x = self.ctx_conv1(out5a_combined)
        x = self.ctx_conv2(x)
        x = self.ctx_conv3(x)
        x = self.ctx_conv4(x)
        ###################
        # enhance decoder #
        ###################
        x = self.bilinear2d(x) #64
        x = self.ctx_conv_up1(x)
        x = self.bilinear2d(x) #128
        x = self.ctx_conv_up2(x)
        x = self.bilinear2d(x) #256
        x = self.ctx_conv_up3(x)
        pred_seg = self.ctx_final(x)
        if self.target_available:
            pred_seg = self.upsample_back(target, pred_seg)
            loss = self.seg_loss(pred_seg, target)
            return loss, pred_seg
        return x

    def forward_detection(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x = self.conv1a(img_batch) # 512x256
        x = self.conv1b(x)   
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) # 256x128
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) # 128x64
        x = self.res3a_branch2b(x)
        out3a_ = self.out3a(x)
        x = self.poolind2d(x)
        x = self.res4a_branch2a(x) # 64x32
        x = self.res4a_branch2b(x)
        # x = self.poolind2d(x)    # only detection do this

        x = self.res5a_branch2a(x) 
        x = self.res5a_branch2b(x)
        x = self.out5a(x)
        out5a_up2 = self.bilinear2d(x)
        out5a_combined = out5a_up2 + out3a_ # 128 x 64

        x = self.ctx_conv1(out5a_combined)
        x = self.ctx_conv2(x)
        x = self.ctx_conv3(x)
        x = self.ctx_conv4(x)                                      #at 128x64 , 64x32 , 32x16 , 16x8 , 8x4
        ###################
        # enhance decoder #
        ###################
        x = self.bilinear2d(x) 
        # x = torch.cat((x, out2),1)
        x = self.ctx_conv_up1(x) # 256x128
        x = self.bilinear2d(x) 
        x = self.ctx_conv_up2(x) # 512x256
        x = self.bilinear2d(x) # 
        x = self.ctx_conv_up3(x) # 1024x512
        pred_seg = self.ctx_final(x)

            
        #############
        # detection #
        #############
        ctx_out1      = self.ctx_conv1_det(out5a_combined) #128x64
        ctx_out1_down = self.poolind2d(ctx_out1)
        ctx_out2      = self.ctx_conv2_det(ctx_out1_down)  #64x32
        ctx_out2_down = self.poolind2d(ctx_out2)
        ctx_out3      = self.ctx_conv3_det(ctx_out2_down) #32x16
        # ctx_out3_down = self.poolind2d(ctx_out3)
        ctx_out4      = self.ctx_conv4_det(ctx_out3) #16x8
        # ctx_out4_down = self.poolind2d(ctx_out4)
        ctx_out5      = self.ctx_conv5_det(ctx_out4) #8x4

        ############################################
        # segmentation output connect to detection #
        ############################################
        pred_seg_down = self.maxpool(pred_seg[:,11:,:,:]) # 512X256
        pred_seg_down = self.maxpool(pred_seg_down) # 256X128
        pred_seg_down = self.maxpool(pred_seg_down) # 128X64
        ctx_out1 = torch.cat([ctx_out1, pred_seg_down], dim = 1)
        pred_seg_down = self.maxpool(pred_seg_down) # 64X32
        ctx_out2 = torch.cat([ctx_out2, pred_seg_down], dim = 1)
        pred_seg_down = self.maxpool(pred_seg_down) # 32X16
        ctx_out3 = torch.cat([ctx_out3, pred_seg_down], dim = 1)

        pred_seg_down = self.maxpool(pred_seg_down) # 16x8
        ctx_out4 = torch.cat([ctx_out4, pred_seg_down], dim = 1)
        pred_seg_down = self.maxpool(pred_seg_down) # 16x8
        ctx_out5 = torch.cat([ctx_out5, pred_seg_down], dim = 1)
            

        regression = torch.cat([self.regressionModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], self.nms_threshold)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
    def forward_lane(self, inputs):
        if self.target_available:
            img_batch, seg_gt, cls_gt = inputs
        else:
            img_batch = inputs
        x = self.conv1a(img_batch) # 1024x 512
        x = self.conv1b(x) 
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) # 512x256
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) # 256x128
        x = self.res3a_branch2b(x)
        out3a_ = self.out3a(x)
        x = self.poolind2d(x)
        x = self.res4a_branch2a(x) # 128x64
        x = self.res4a_branch2b(x)
        # x = self.poolind2d(x)

        x = self.res5a_branch2a(x) # 64x32
        x = self.res5a_branch2b(x)
        x = self.out5a(x)
        # out5a_up2 = self.Bilinear_Interpolation(x)
        out5a_up2 = self.bilinear2d(x)
        # print(out3a_.size())
        # print(out5a_up2.size())
        out5a_combined = out5a_up2 + out3a_

        x = self.ctx_conv1(out5a_combined)
        x = self.ctx_conv2(x)
        x = self.ctx_conv3(x)
        x = self.ctx_conv4(x)
        ########
        # lane #
        ########
        x = self.bilinear2d(x)  # 256x128
        x = self.ctx_conv_up1_1_lane(x)
        x = self.ctx_conv_up1_2_lane(x)
        x = self.ctx_conv_up1_3_lane(x)
        x = self.ctx_conv_up1_4_lane(x)
        x = self.ctx_conv_up1_5_lane(x)
        x = self.bilinear2d(x)  # 512x256
        x = self.ctx_conv_up2_1_lane(x)
        x = self.ctx_conv_up2_2_lane(x)
        x = self.ctx_conv_up2_3_lane(x)
        x = self.ctx_conv_up2_4_lane(x)
        x = self.ctx_conv_up2_5_lane(x)
        x = self.bilinear2d(x)  # 1024x512
        # segmenation output
        out_temp = self.ctx_conv_up3_lane(x)
        pred_seg = self.ctx_final_lane(out_temp)
        # classification output
        ##########################################
        # maybe can use intermediate supervision #
        ##########################################
        pred_cls = self.ctx_conv_up3_class(out_temp)
        pred_cls = self.avgpool(pred_cls[:,1:,:,:])
        # print(pred_cls.size())
        pred_cls = pred_cls.squeeze(dim = 2)
        pred_cls = pred_cls.squeeze(dim = 2)
        pred_cls = self.sigmoid(pred_cls)
        # print("pred_cls.size()", pred_cls.size())
        # print("pred_seg.size()", pred_seg.size())
        if self.target_available:
            # segmentation
            pred_seg = self.upsample_back(seg_gt, pred_seg)
            loss_seg = self.lane_loss(pred_seg, seg_gt)
            # classification
            loss_cls = self.binary_class_loss(pred_cls, cls_gt)
            return loss_seg, pred_seg, loss_cls, pred_cls      
        return pred_seg, pred_cls

        """
        ########
        # lane #
        ########
        x = self.bilinear2d(x)  # 256x128
        x = self.ctx_conv_up1_lane(x)
        x = self.bilinear2d(x)  # 512x256
        x = self.ctx_conv_up2_lane(x)
        x = self.bilinear2d(x)  # 1024x512
        # segmenation output
        out_temp = self.ctx_conv_up3_lane(x)
        pred_seg = self.ctx_final_lane(out_temp)
        # classification output
        ##########################################
        # maybe can use intermediate supervision #
        ##########################################
        pred_cls = self.ctx_conv_up3_class(out_temp)
        pred_cls = self.avgpool(pred_cls[:,1:,:,:])
        # print(pred_cls.size())
        pred_cls = pred_cls.squeeze(dim = 2)
        pred_cls = pred_cls.squeeze(dim = 2)
        pred_cls = self.sigmoid(pred_cls)

        ########
        # lane #
        ########
        x = self.bilinear2d(x)  # 256x128
        x = self.ctx_conv_up1_lane(x)
        x = self.bilinear2d(x)  # 512x256
        x = self.ctx_conv_up2_lane(x)
        x = self.bilinear2d(x)  # 1024x512
        # segmenation output
        x = self.ctx_conv_up3_lane(x)
        pred_seg = self.ctx_final_lane(x)
        # classification output
        ##########################################
        # maybe can use intermediate supervision #
        ##########################################
        out_temp = self.softmax(pred_seg)
        pred_cls = self.ctx_conv_up3_class(out_temp[:,1:,:,:])
        pred_cls = self.avgpool(pred_cls)
        # print(pred_cls.size())
        pred_cls = pred_cls.squeeze(dim = 2)
        pred_cls = pred_cls.squeeze(dim = 2)
        pred_cls = self.sigmoid(pred_cls)
        """