
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.net_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes, RegressionModel, ClassificationModel
from network.anchors import Anchors
from network.anchors_ssd import Anchors_SSD
from network.loss_seg import CrossEntropyLoss2d 
import network.losses as losses
import time
# for visualize
import numpy as np
import cv2

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


class Jacinto_detseg_256x512_v1(nn.Module):
    def __init__(self, num_classes_seg, num_classes_OD, use_focal_loss, pretrained=None, mode = "detection", normalize_anchor = False):
        super(Jacinto_detseg_256x512_v1, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.normalize_anchor = normalize_anchor
        self.num_classes_seg = num_classes_seg
        self.num_classes_OD = num_classes_OD -1 if self.use_focal_loss else num_classes_OD
        # self.num_classes_OD = num_classes_OD
        self.pretrained = pretrained
        self.mode = mode
        self.target_available = True
        self.nms_threshold = 0.4
        num_anchors = 6
        print("self.num_classes_OD", self.num_classes_OD)
        self.use_ctx_final = True
        # self.num_class = num_classes
        # self.input_size = input_size
        # self.phase = phase

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
        if self.use_ctx_final:
            self.ctx_final = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1))
        else:
            self.ctx_final_ = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1))
            # self.ctx_final_ = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1),
                            # nn.ReLU())
        # self.seg_loss = CrossEntropyLoss2d(weight = True)
        self.seg_loss = CrossEntropyLoss2d()
        #############
        # detection #
        #############
        self.ctx_conv1_det = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv2_det = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv3_det = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv4_det = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1, 1 ,1),nn.ReLU())
        self.ctx_conv5_det = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1, 1 ,1),nn.ReLU())

        self.regressionModel = RegressionModel(64, num_anchors = num_anchors)
        self.classificationModel = ClassificationModel(64, num_anchors = num_anchors, num_classes= self.num_classes_OD)

        # self.anchors = Anchors()
        pyramid_levels = [3, 4, 5, 6, 7]
        min_sizes  = [20.48, 25.60, 51.20, 76.30, 128.0 ]
        max_sizes  = [51.20, 51.20, 76.80, 128.0, 176.17]
        offsets    = [  0.5,   0.5, 0.5,  0.5,   0.5]
        ratios = [1, 1, 0.5, 2, 0.3333, 3]
        self.anchors = Anchors_SSD(normalize_anchor = self.normalize_anchor, min_sizes = min_sizes, max_sizes = max_sizes,\
                                   offsets = offsets, ratios = ratios, pyramid_levels = pyramid_levels)
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()   
        self.focalLoss = losses.FocalLoss()

        #############################
        # up and downsample formula #
        #############################
        self.poolind2d = nn.MaxPool2d(2,2)
        # self.bilinear2d =nn.Upsample(scale_factor=2, mode='nearest')
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # if self.pretrained:
        #     assert os.path.exists(self.pretrained),"Pretrain model {} does not exist.".format(self.pretrained)
        #     loaded_state = torch.load(self.pretrained)
        #     net.load_state_dict(loaded_state["model_state"], strict=False)
        #     print("Resume from pretrain model {}".format(self.pretrained))

    def freeze_bn_layers(self):
        '''Freeze BatchNorm running statictic.'''
        for name, module in self.named_children():
            if name in self.bn_in_layers:
                for layer in module:
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.eval()

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
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def forward(self, inputs):
        if self.mode == "segmentation":
            return self.forward_segmentation(inputs)
        elif self.mode == "detection":
            return self.forward_detection(inputs)
        elif self.mode == "lane":
           return self.forward_lane_bdd(inputs)
        #elif self.mode == "lane_line":
        #    return self.forward_lane_line_bdd(inputs)
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
            if self.use_ctx_final:
                pred_seg = self.ctx_final(x)
            else:
                pred_seg = self.ctx_final_(x)
            # pred_seg = self.ctx_final_(x)

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

            regression = torch.cat([self.regressionModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
    
            anchors = self.anchors(img_batch)
            # if self.training and self.target_available:
            if self.target_available:
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
                scores_over_thresh = (scores>0.3)[0, :, 0]
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
                    if len(transformed_anchors.size()) < 2:
                        return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), pred_seg]                    
                    anchors_nms_idx = nms(transformed_anchors, scores, self.nms_threshold)
                toc = time.time()
                print("NMS took {}".format(toc-tic))
                transformed_anchors = transformed_anchors.squeeze()
                nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
                # segmentation
                pred_seg = pred_seg #8x downsample
                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
                try:
                    return [nms_scores, nms_class, transformed_anchors[anchors_nms_idx, :], pred_seg]
                except:
                    return [nms_scores, nms_class, transformed_anchors, pred_seg]

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
        if self.use_ctx_final:
            pred_seg = self.ctx_final(x)
        else:
            pred_seg = self.ctx_final_(x)
        # pred_seg = self.ctx_final_(x)
        # pred_seg = self.softmax(pred_seg)

        # seg_out = self.softmax(pred_seg)
        # seg_out = seg_out[:, 1:, :, :]
        # print("seg_out.size()", seg_out.size())
        # seg_out = seg_out.max(1)[0]
        # print(seg_out.size())
        # seg_out = seg_out.data.cpu().numpy()
        # print(np.max(seg_out), np.min(seg_out))
        # seg_out = np.squeeze(seg_out, axis = 0)
        # seg_out[seg_out > 0.5]  = 255
        # seg_out = seg_out.astype(np.uint8)
        # cv2.imshow("seg_out", seg_out)
        # cv2.waitKey(0)
        if self.target_available:
            pred_seg = self.upsample_back(target, pred_seg)
            loss = self.seg_loss(pred_seg, target)
            return loss, pred_seg
        else:
            return pred_seg

    def forward_detection(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        # print(img_batch.size())
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
        # print("out5a_up2", out5a_up2.size())
        out5a_combined = out5a_up2 + out3a_ # 128 x 64

        # x = self.ctx_conv1(out5a_combined)
        # x = self.ctx_conv2(x)
        # x = self.ctx_conv3(x)
        # x = self.ctx_conv4(x)                                      #at 128x64 , 64x32 , 32x16 , 16x8 , 8x4
        # ###################
        # # enhance decoder #
        # ###################
        # x = self.bilinear2d(x) 
        # # x = torch.cat((x, out2),1)
        # x = self.ctx_conv_up1(x) # 256x128
        # x = self.bilinear2d(x) 
        # x = self.ctx_conv_up2(x) # 512x256
        # x = self.bilinear2d(x) # 
        # x = self.ctx_conv_up3(x) # 1024x512
        # if self.use_ctx_final:
        #     x = self.ctx_final(x)
        # else:
        #     x = self.ctx_final_(x)

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
        # print(ctx_out1.size())
        # print(ctx_out2.size())
        # print(ctx_out3.size())
        # print(ctx_out4.size())
        # print(ctx_out5.size())
        regression = torch.cat([self.regressionModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss([classification, regression, anchors], annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.3)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            tic = time.time()
            if torch.__version__[0] == "0":
                anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], self.nms_threshold)
            else:
                transformed_anchors = torch.squeeze(transformed_anchors)
                scores = torch.squeeze(scores)
                if len(transformed_anchors.size()) < 2:
                    return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]                    
                anchors_nms_idx = nms(transformed_anchors, scores, self.nms_threshold)
            toc = time.time()
            print("NMS took {}".format(toc-tic))
            transformed_anchors = transformed_anchors.squeeze()
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            try:
                return [nms_scores, nms_class, transformed_anchors[anchors_nms_idx, :]]
            except:
                return [nms_scores, nms_class, transformed_anchors]

























##############################################
# adopt retinanet anchor with scale x+1 double
#############################################
class Jacinto_detseg_256x512_v4(nn.Module):
    def __init__(self, num_classes_seg, num_classes_OD, use_focal_loss, pretrained=None, mode = "detection", normalize_anchor = False):
        super(Jacinto_detseg_256x512_v4, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.normalize_anchor = normalize_anchor
        self.num_classes_seg = num_classes_seg
        self.num_classes_OD = num_classes_OD -1 if self.use_focal_loss else num_classes_OD
        self.pretrained = pretrained
        self.mode = mode
        self.target_available = True
        self.nms_threshold = 0.4
        num_anchors = 15
        print("self.num_classes_OD", self.num_classes_OD)
        self.use_ctx_final = True
        # self.num_class = num_classes
        # self.input_size = input_size
        # self.phase = phase

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
        if self.use_ctx_final:
            self.ctx_final = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1))
        else:
            self.ctx_final_ = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1))
            # self.ctx_final_ = nn.Sequential(nn.Conv2d(64, self.num_classes_seg, 3, 1, 1, 1 ,1),
                            # nn.ReLU())
        # self.seg_loss = CrossEntropyLoss2d(weight = True)
        self.seg_loss = CrossEntropyLoss2d()
        #############
        # detection #
        #############
        self.ctx_conv1_det = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv1_det_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv2_det = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv2_det_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv3_det = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv3_det_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv4_det = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1, 1 ,1),nn.ReLU())
        self.ctx_conv4_det_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())
        self.ctx_conv5_det = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1, 1 ,1),nn.ReLU())
        self.ctx_conv5_det_2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, 1 ,1),nn.ReLU())

        self.regressionModel = RegressionModel(64, num_anchors = num_anchors)
        self.classificationModel = ClassificationModel(64, num_anchors = num_anchors, num_classes= self.num_classes_OD)

        # self.anchors = Anchors()
        # pyramid_levels = [3, 4, 5, 6, 7]
        # min_sizes  = [20.48, 25.60, 51.20, 76.30, 128.0 ]
        # max_sizes  = [51.20, 51.20, 76.80, 128.0, 176.17]
        # offsets    = [  0.5,   0.5, 0.5,  0.5,   0.5]
        # ratios = [1, 1, 0.5, 2, 0.3333, 3]
        # self.anchors = Anchors_SSD(normalize_anchor = self.normalize_anchor, min_sizes = min_sizes, max_sizes = max_sizes,\
        #                            offsets = offsets, ratios = ratios, pyramid_levels = pyramid_levels)

        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0), 2 ** (-1.0 / 3.0), 2 ** (-2.0 / 3.0)])
        self.anchors = Anchors(scales = scales, double = True)
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()   
        self.focalLoss = losses.FocalLoss()

        #############################
        # up and downsample formula #
        #############################
        self.poolind2d = nn.MaxPool2d(2,2)
        # self.bilinear2d =nn.Upsample(scale_factor=2, mode='nearest')
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # if self.pretrained:
        #     assert os.path.exists(self.pretrained),"Pretrain model {} does not exist.".format(self.pretrained)
        #     loaded_state = torch.load(self.pretrained)
        #     net.load_state_dict(loaded_state["model_state"], strict=False)
        #     print("Resume from pretrain model {}".format(self.pretrained))

    def freeze_bn_layers(self):
        '''Freeze BatchNorm running statictic.'''
        for name, module in self.named_children():
            if name in self.bn_in_layers:
                for layer in module:
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.eval()

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
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def forward(self, inputs):
        if self.mode == "segmentation":
            return self.forward_segmentation(inputs)
        elif self.mode == "detection":
            return self.forward_detection(inputs)
        elif self.mode == "lane":
           return self.forward_lane_bdd(inputs)
        #elif self.mode == "lane_line":
        #    return self.forward_lane_line_bdd(inputs)
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
            if self.use_ctx_final:
                pred_seg = self.ctx_final(x)
            else:
                pred_seg = self.ctx_final_(x)
            # pred_seg = self.ctx_final_(x)

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

            regression = torch.cat([self.regressionModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in [ctx_out1, ctx_out2, ctx_out3, ctx_out4, ctx_out5]], dim=1)
    
            anchors = self.anchors(img_batch)
            # if self.training and self.target_available:
            if self.target_available:
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
                scores_over_thresh = (scores>0.3)[0, :, 0]
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
                    if len(transformed_anchors.size()) < 2:
                        return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), pred_seg]                    
                    anchors_nms_idx = nms(transformed_anchors, scores, self.nms_threshold)
                toc = time.time()
                print("NMS took {}".format(toc-tic))
                transformed_anchors = transformed_anchors.squeeze()
                nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
                # segmentation
                pred_seg = pred_seg #8x downsample
                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
                try:
                    return [nms_scores, nms_class, transformed_anchors[anchors_nms_idx, :], pred_seg]
                except:
                    return [nms_scores, nms_class, transformed_anchors, pred_seg]

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
        if self.use_ctx_final:
            pred_seg = self.ctx_final(x)
        else:
            pred_seg = self.ctx_final_(x)
        # pred_seg = self.ctx_final_(x)
        # pred_seg = self.softmax(pred_seg)

        # seg_out = self.softmax(pred_seg)
        # seg_out = seg_out[:, 1:, :, :]
        # print("seg_out.size()", seg_out.size())
        # seg_out = seg_out.max(1)[0]
        # print(seg_out.size())
        # seg_out = seg_out.data.cpu().numpy()
        # print(np.max(seg_out), np.min(seg_out))
        # seg_out = np.squeeze(seg_out, axis = 0)
        # seg_out[seg_out > 0.5]  = 255
        # seg_out = seg_out.astype(np.uint8)
        # cv2.imshow("seg_out", seg_out)
        # cv2.waitKey(0)
        if self.target_available:
            pred_seg = self.upsample_back(target, pred_seg)
            loss = self.seg_loss(pred_seg, target)
            return loss, pred_seg
        else:
            return pred_seg

    def forward_detection(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        # print(img_batch.size())
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
        # print("out5a_up2", out5a_up2.size())
        out5a_combined = out5a_up2 + out3a_ # 128 x 64

        # x = self.ctx_conv1(out5a_combined)
        # x = self.ctx_conv2(x)
        # x = self.ctx_conv3(x)
        # x = self.ctx_conv4(x)                                      #at 128x64 , 64x32 , 32x16 , 16x8 , 8x4
        # ###################
        # # enhance decoder #
        # ###################
        # x = self.bilinear2d(x) 
        # # x = torch.cat((x, out2),1)
        # x = self.ctx_conv_up1(x) # 256x128
        # x = self.bilinear2d(x) 
        # x = self.ctx_conv_up2(x) # 512x256
        # x = self.bilinear2d(x) # 
        # x = self.ctx_conv_up3(x) # 1024x512
        # if self.use_ctx_final:
        #     x = self.ctx_final(x)
        # else:
        #     x = self.ctx_final_(x)

        #############
        # detection #
        #############
        ctx_out1      = self.ctx_conv1_det(out5a_combined) #128x64
        ctx_out1_1      = self.ctx_conv1_det_2(ctx_out1) #128x64
        ctx_out1_down = self.poolind2d(ctx_out1)

        ctx_out2      = self.ctx_conv2_det(ctx_out1_down)  #64x32
        ctx_out2_1      = self.ctx_conv2_det_2(ctx_out2) 
        ctx_out2_down = self.poolind2d(ctx_out2)

        ctx_out3      = self.ctx_conv3_det(ctx_out2_down) #32x16
        ctx_out3_1      = self.ctx_conv3_det_2(ctx_out3) 
        
        ctx_out4      = self.ctx_conv4_det(ctx_out3) #16x8
        ctx_out4_1      = self.ctx_conv4_det_2(ctx_out4) 
        
        ctx_out5      = self.ctx_conv5_det(ctx_out4) #8x4
        ctx_out5_1      = self.ctx_conv5_det_2(ctx_out5) 
        # print(ctx_out1.size())
        # print(ctx_out2.size())
        # print(ctx_out3.size())
        # print(ctx_out4.size())
        # print(ctx_out5.size())
        regression = torch.cat([self.regressionModel(feature) for feature in [ctx_out1, ctx_out1_1, ctx_out2, ctx_out2_1, ctx_out3, ctx_out3_1, ctx_out4, ctx_out4_1, ctx_out5, ctx_out5_1]], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in [ctx_out1, ctx_out1_1, ctx_out2, ctx_out2_1, ctx_out3, ctx_out3_1, ctx_out4, ctx_out4_1, ctx_out5, ctx_out5_1]], dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss([classification, regression, anchors], annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.3)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            tic = time.time()
            if torch.__version__[0] == "0":
                anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], self.nms_threshold)
            else:
                transformed_anchors = torch.squeeze(transformed_anchors)
                scores = torch.squeeze(scores)
                if len(transformed_anchors.size()) < 2:
                    return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]                    
                anchors_nms_idx = nms(transformed_anchors, scores, self.nms_threshold)
            toc = time.time()
            print("NMS took {}".format(toc-tic))
            transformed_anchors = transformed_anchors.squeeze()
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            try:
                return [nms_scores, nms_class, transformed_anchors[anchors_nms_idx, :]]
            except:
                return [nms_scores, nms_class, transformed_anchors]