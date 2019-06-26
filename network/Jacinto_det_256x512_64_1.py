#################################
# Jacinto net detection model
# 8head 
# training on IVS_SUR_dataset
#################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.net_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes, RegressionModel, ClassificationModel
from network.anchors_ssd import Anchors_SSD
import network.losses as losses
from network.multibox_loss import MultiBoxLoss
import time

if torch.__version__[0] == "0":
    from lib.nms.pth_nms import pth_nms as nms_lib
    def nms(dets, thresh):
        "Dispatch to either CPU or GPU NMS implementations.\
        <Acc></Acc>ept dets as tensor"""
        return nms_lib(dets, thresh)
elif torch.__version__[0] == "1":
    from network.nms import boxes_nms as nms

class JacintoNet_det_256x512_64_1(nn.Module):
    def __init__(self, num_classes_OD = 4, normalize_anchor = False):
        super(JacintoNet_det_256x512_64_1, self).__init__()
        self.num_classes_OD = num_classes_OD
        self.num_anchors = 6
        self.num_conf_channel = self.num_anchors * self.num_classes_OD
        self.num_loc_channel = self.num_anchors * 4
        self.nms_threshold = 0.4
        self.normalize_anchor = normalize_anchor
        self.use_conf = True

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
        
        self.ctx_output1    = self._make_conv_block(128, 256, 1, 1, 0, 1, 1)
        self.ctx_output32   = self._make_conv_block(256, 256, 1, 1, 0, 1, 1)
        self.ctx_output32_ = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output2    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)
        self.ctx_output2_  = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output3    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)
        self.ctx_output3_  = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output4    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)

        ###############################
        # location and classification #
        ###############################
        self.ctx_output1_relu_mbox_loc      = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output1_relu_mbox_conf    = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output1_relu_mbox_conf_    = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output32_relu_mbox_loc     = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output32_relu_mbox_conf   = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output32_relu_mbox_conf_   = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output32__relu_mbox_loc    = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output32__relu_mbox_conf  = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output32__relu_mbox_conf_  = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output2_relu_mbox_loc     = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output2_relu_mbox_conf   = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output2_relu_mbox_conf_   = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output2__relu_mbox_loc   = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output2__relu_mbox_conf = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output2__relu_mbox_conf_ = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output3_relu_mbox_loc     = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output3_relu_mbox_conf   = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output3_relu_mbox_conf_   = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output3__relu_mbox_loc    = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output3__relu_mbox_conf  = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output3__relu_mbox_conf_  = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        self.ctx_output4_relu_mbox_loc      = nn.Conv2d(256, self.num_loc_channel, 3, 1, 1, 1 ,1)
        if self.use_conf:
            self.ctx_output4_relu_mbox_conf    = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)
        else:
            self.ctx_output4_relu_mbox_conf_    = nn.Conv2d(256, self.num_conf_channel, 3, 1, 1, 1 ,1)

        
        self.poolind2d = nn.MaxPool2d(2,2)
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')

        pyramid_levels = [3, 4, 4, 5, 5, 6, 6, 7]
        min_sizes  = [20.48, 25.60, 25.60, 51.20, 51.20, 76.30, 76.30, 128.0 ]
        max_sizes  = [51.20, 51.20, 51.20, 76.80, 76.80, 128.0, 128.0, 176.17]
        offsets    = [  0.5,   0.5,   1.0,   0.5,   1.0,   0.5,   1.0,   0.5]
        ratios = [1, 1, 0.5, 2, 0.3333, 3]
        self.anchors = Anchors_SSD(normalize_anchor = self.normalize_anchor, min_sizes = min_sizes, max_sizes = max_sizes,\
                                   offsets = offsets, ratios = ratios, pyramid_levels = pyramid_levels)
        self.regressBoxes = BBoxTransform(normalize_coor = self.normalize_anchor)
        self.clipBoxes = ClipBoxes()   
        self.softmax = nn.Softmax(dim = 2)
        self.multiboxloss = MultiBoxLoss(self.num_classes_OD, 0.5, True, 0, True, 6, 0.5, False, True)

    def _make_conv_block(self, in_channel, out_channel, kernel_size, stride, padding, dilation = 1, group = 1):
        layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation ,group),
                               nn.BatchNorm2d(out_channel),
                            nn.ReLU()
        )
        return layers

    def loc_permute(self, loc_tensor):
        loc_tensor = loc_tensor.permute(0, 2, 3, 1)
        return loc_tensor.contiguous().view(loc_tensor.shape[0], -1, 4)

    def conf_permute(self, conf_tensor):
        # out is B x C x W x H, with C = n_classes + n_anchors
        # print("conf_tensor.size()", conf_tensor.size())
        conf_tensor_per = conf_tensor.permute(0, 2, 3, 1)
        batch_size, width, height, channels = conf_tensor_per.shape
        conf_tensor_per = conf_tensor_per.view(batch_size, width, height, self.num_anchors, self.num_classes_OD)
        return conf_tensor_per.contiguous().view(conf_tensor.shape[0], -1, self.num_classes_OD)

    def forward(self, inputs):
        """ SSD forward pass"""
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        # print("img_batch.size()", img_batch.size())
        x = self.conv1a(img_batch)
        x = self.conv1b(x) #256
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) #128
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) #64
        res3_out = self.res3a_branch2b(x)
        x = self.poolind2d(res3_out)
        x = self.res4a_branch2a(x) #32
        res4_out = self.res4a_branch2b(x)
        x = self.poolind2d(res4_out)
        x = self.res5a_branch2a(x) #16
        res5_out = self.res5a_branch2b(x)
        pool6 = self.poolind2d(res5_out)
        pool7 = self.poolind2d(pool6)

        ctx_output1   = self.ctx_output1(res3_out)
        ctx_output32  = self.ctx_output32(res4_out)
        ctx_output32_ = self.ctx_output32_(ctx_output32)
        ctx_output2   = self.ctx_output2(res5_out)
        ctx_output2_  = self.ctx_output2_(ctx_output2)
        ctx_output3  = self.ctx_output3(pool6)
        ctx_output3_ = self.ctx_output3_(ctx_output3)
        ctx_output4 = self.ctx_output4(pool7)

        ctx_loc1  = self.ctx_output1_relu_mbox_loc(ctx_output1)
        if self.use_conf:
            ctx_conf1 = self.ctx_output1_relu_mbox_conf(ctx_output1)
        else:
            ctx_conf1 = self.ctx_output1_relu_mbox_conf_(ctx_output1)
        # permute
        ctx_loc1 = self.loc_permute(ctx_loc1)
        ctx_conf1 = self.conf_permute(ctx_conf1)

        ctx_loc32  = self.ctx_output32_relu_mbox_loc(ctx_output32)
        if self.use_conf:
            ctx_conf32 = self.ctx_output32_relu_mbox_conf(ctx_output32)
        else:
            ctx_conf32 = self.ctx_output32_relu_mbox_conf_(ctx_output32)
        # permute
        ctx_loc32 = self.loc_permute(ctx_loc32)
        ctx_conf32 = self.conf_permute(ctx_conf32)

        ctx_loc32_  = self.ctx_output32__relu_mbox_loc(ctx_output32_)
        if self.use_conf:
            ctx_conf32_ = self.ctx_output32__relu_mbox_conf(ctx_output32_)
        else:
            ctx_conf32_ = self.ctx_output32__relu_mbox_conf_(ctx_output32_)
        # permute
        ctx_loc32_ = self.loc_permute(ctx_loc32_)
        ctx_conf32_ = self.conf_permute(ctx_conf32_)

        ctx_loc2  = self.ctx_output2_relu_mbox_loc(ctx_output2)
        if self.use_conf:
            ctx_conf2 = self.ctx_output2_relu_mbox_conf(ctx_output2)
        else:
            ctx_conf2 = self.ctx_output2_relu_mbox_conf_(ctx_output2)
        # permute
        ctx_loc2 = self.loc_permute(ctx_loc2)
        ctx_conf2 = self.conf_permute(ctx_conf2)

        ctx_loc2_  = self.ctx_output2__relu_mbox_loc(ctx_output2_)
        if self.use_conf:
            ctx_conf2_ = self.ctx_output2__relu_mbox_conf(ctx_output2_)
        else:
            ctx_conf2_ = self.ctx_output2__relu_mbox_conf_(ctx_output2_)
        # permute
        ctx_loc2_ = self.loc_permute(ctx_loc2_)
        ctx_conf2_ = self.conf_permute(ctx_conf2_)

        ctx_loc3  = self.ctx_output3_relu_mbox_loc(ctx_output3)
        if self.use_conf:
            ctx_conf3 = self.ctx_output3_relu_mbox_conf(ctx_output3)
        else:
            ctx_conf3 = self.ctx_output3_relu_mbox_conf_(ctx_output3)
        # permute
        ctx_loc3 = self.loc_permute(ctx_loc3)
        ctx_conf3 = self.conf_permute(ctx_conf3)

        ctx_loc3_  = self.ctx_output3__relu_mbox_loc(ctx_output3_)
        if self.use_conf:
            ctx_conf3_ = self.ctx_output3__relu_mbox_conf(ctx_output3_)
        else:
            ctx_conf3_ = self.ctx_output3__relu_mbox_conf_(ctx_output3_)
        # permute
        ctx_loc3_ = self.loc_permute(ctx_loc3_)
        ctx_conf3_ = self.conf_permute(ctx_conf3_)
     
        ctx_loc4  = self.ctx_output4_relu_mbox_loc(ctx_output4)
        if self.use_conf:
            ctx_conf4 = self.ctx_output4_relu_mbox_conf(ctx_output4)
        else:
            ctx_conf4 = self.ctx_output4_relu_mbox_conf_(ctx_output4)
        # permute
        ctx_loc4 = self.loc_permute(ctx_loc4)
        ctx_conf4 = self.conf_permute(ctx_conf4)
        # print("ctx_conf4.size()", ctx_conf4.size())

        regression     = torch.cat([ctx_loc1, ctx_loc32, ctx_loc32_, ctx_loc2, ctx_loc2_, ctx_loc3, ctx_loc3_, ctx_loc4], dim=1)
        classification = torch.cat([ctx_conf1, ctx_conf32, ctx_conf32_, ctx_conf2, ctx_conf2_, ctx_conf3, ctx_conf3_, ctx_conf4], dim=1)
        anchors = self.anchors(img_batch)
        # print("regression.size()", regression.size())
        if self.training:
            loss_l, loss_c = self.multiboxloss([regression, classification, anchors], annotations)
            # print(loss_l, loss_c)
            return loss_c, loss_l
        else:
            # print(classification.size())
            classification = self.softmax(classification)
            classification = classification[:,:,1:]
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
            # segmentation
            # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
            # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]
            try:
                return [nms_scores, nms_class, transformed_anchors[anchors_nms_idx, :]]
            except:
                return [nms_scores, nms_class, transformed_anchors]







class JacintoNet_det_256x512_64_1_retinanet(nn.Module):
    def __init__(self, num_classes_OD = 4):
        super(JacintoNet_det_256x512_64_1_retinanet, self).__init__()
        self.num_classes_OD = num_classes_OD
        self.num_anchors = 6
        self.nms_threshold = 0.4

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
        
        self.ctx_output1    = self._make_conv_block(128, 256, 1, 1, 0, 1, 1)
        self.ctx_output32   = self._make_conv_block(256, 256, 1, 1, 0, 1, 1)
        self.ctx_output32_ = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output2    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)
        self.ctx_output2_  = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output3    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)
        self.ctx_output3_  = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output4    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)

        ###############################
        # location and classification #
        ###############################
        self.ctx_output1_relu_mbox_loc     = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output1_relu_mbox_conf_    = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output32_relu_mbox_loc    = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output32_relu_mbox_conf_   = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output32__relu_mbox_loc  = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output32__relu_mbox_conf_ = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output2_relu_mbox_loc     = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output2_relu_mbox_conf_  = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output2__relu_mbox_loc  = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output2__relu_mbox_conf_ = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output3_relu_mbox_loc     = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output3_relu_mbox_conf_  = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output3__relu_mbox_loc     = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output3__relu_mbox_conf_  = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.ctx_output4_relu_mbox_loc    = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)
        self.ctx_output4_relu_mbox_conf_    = nn.Conv2d(256, 24, 3, 1, 1, 1 ,1)

        self.poolind2d = nn.MaxPool2d(2,2)
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')

        self.anchors = Anchors_SSD()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()   
        self.output_act = nn.Sigmoid()
        self.focalLoss = losses.FocalLoss()
        # self.softmax = nn.Softmax(dim = 2)

    def _make_conv_block(self, in_channel, out_channel, kernel_size, stride, padding, dilation = 1, group = 1):
        layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation ,group),
                               nn.BatchNorm2d(out_channel),
                            nn.ReLU()
        )
        return layers

    def loc_permute(self, loc_tensor):
        loc_tensor = loc_tensor.permute(0, 2, 3, 1)
        return loc_tensor.contiguous().view(loc_tensor.shape[0], -1, 4)

    def conf_permute(self, conf_tensor):
        # out is B x C x W x H, with C = n_classes + n_anchors
        conf_tensor_per = conf_tensor.permute(0, 2, 3, 1)
        batch_size, width, height, channels = conf_tensor_per.shape
        conf_tensor_per = conf_tensor_per.view(batch_size, width, height, self.num_anchors, self.num_classes_OD)
        return conf_tensor_per.contiguous().view(conf_tensor.shape[0], -1, self.num_classes_OD)

    def conf_permute_with_sigmoid(self, conf_tensor):
        # out is B x C x W x H, with C = n_classes + n_anchors
        conf_tensor = self.output_act(conf_tensor)
        conf_tensor_per = conf_tensor.permute(0, 2, 3, 1)
        batch_size, width, height, channels = conf_tensor_per.shape
        conf_tensor_per = conf_tensor_per.view(batch_size, width, height, self.num_anchors, self.num_classes_OD)
        return conf_tensor_per.contiguous().view(conf_tensor.shape[0], -1, self.num_classes_OD)

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x = self.conv1a(img_batch)
        x = self.conv1b(x) #256
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) #128
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) #64
        res3_out = self.res3a_branch2b(x)
        x = self.poolind2d(res3_out)
        x = self.res4a_branch2a(x) #32
        res4_out = self.res4a_branch2b(x)
        x = self.poolind2d(res4_out)
        x = self.res5a_branch2a(x) #16
        res5_out = self.res5a_branch2b(x)
        pool6 = self.poolind2d(res5_out)
        pool7 = self.poolind2d(pool6)

        ctx_output1   = self.ctx_output1(res3_out)
        ctx_output32  = self.ctx_output32(res4_out)
        ctx_output32_ = self.ctx_output32_(ctx_output32)
        ctx_output2   = self.ctx_output2(res5_out)
        ctx_output2_  = self.ctx_output2_(ctx_output2)
        ctx_output3  = self.ctx_output3(pool6)
        ctx_output3_ = self.ctx_output3_(ctx_output3)
        ctx_output4 = self.ctx_output4(pool7)

        ctx_loc1  = self.ctx_output1_relu_mbox_loc(ctx_output1)
        ctx_conf1 = self.ctx_output1_relu_mbox_conf_(ctx_output1)
        # permute
        ctx_loc1 = self.loc_permute(ctx_loc1)
        ctx_conf1 = self.conf_permute_with_sigmoid(ctx_conf1)

        ctx_loc32  = self.ctx_output32_relu_mbox_loc(ctx_output32)
        ctx_conf32 = self.ctx_output32_relu_mbox_conf_(ctx_output32)
        # permute
        ctx_loc32 = self.loc_permute(ctx_loc32)
        ctx_conf32 = self.conf_permute_with_sigmoid(ctx_conf32)

        ctx_loc32_  = self.ctx_output32__relu_mbox_loc(ctx_output32_)
        ctx_conf32_ = self.ctx_output32__relu_mbox_conf_(ctx_output32_)
        # permute
        ctx_loc32_ = self.loc_permute(ctx_loc32_)
        ctx_conf32_ = self.conf_permute_with_sigmoid(ctx_conf32_)

        ctx_loc2  = self.ctx_output2_relu_mbox_loc(ctx_output2)
        ctx_conf2 = self.ctx_output2_relu_mbox_conf_(ctx_output2)
        # permute
        ctx_loc2 = self.loc_permute(ctx_loc2)
        ctx_conf2 = self.conf_permute_with_sigmoid(ctx_conf2)

        ctx_loc2_  = self.ctx_output2__relu_mbox_loc(ctx_output2_)
        ctx_conf2_ = self.ctx_output2__relu_mbox_conf_(ctx_output2_)
        # permute
        ctx_loc2_ = self.loc_permute(ctx_loc2_)
        ctx_conf2_ = self.conf_permute_with_sigmoid(ctx_conf2_)

        ctx_loc3  = self.ctx_output3_relu_mbox_loc(ctx_output3)
        ctx_conf3 = self.ctx_output3_relu_mbox_conf_(ctx_output3)
        # permute
        ctx_loc3 = self.loc_permute(ctx_loc3)
        ctx_conf3 = self.conf_permute_with_sigmoid(ctx_conf3)

        ctx_loc3_  = self.ctx_output3__relu_mbox_loc(ctx_output3_)
        ctx_conf3_ = self.ctx_output3__relu_mbox_conf_(ctx_output3_)
        # permute
        ctx_loc3_ = self.loc_permute(ctx_loc3_)
        ctx_conf3_ = self.conf_permute_with_sigmoid(ctx_conf3_)
     
        ctx_loc4  = self.ctx_output4_relu_mbox_loc(ctx_output4)
        ctx_conf4 = self.ctx_output4_relu_mbox_conf_(ctx_output4)
        # permute
        ctx_loc4 = self.loc_permute(ctx_loc4)
        ctx_conf4 = self.conf_permute_with_sigmoid(ctx_conf4)
        # print(ctx_loc1.size())

        regression     = torch.cat([ctx_loc1, ctx_loc32, ctx_loc32_, ctx_loc2, ctx_loc2_, ctx_loc3, ctx_loc3_, ctx_loc4], dim=1)
        classification = torch.cat([ctx_conf1, ctx_conf32, ctx_conf32_, ctx_conf2, ctx_conf2_, ctx_conf3, ctx_conf3_, ctx_conf4], dim=1)
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




class JacintoNet_det_256x512_64_1_retinanet_only_encoder(nn.Module):
    """Only use backbone of the model"""
    def __init__(self, num_classes_OD = 3):
        super(JacintoNet_det_256x512_64_1_retinanet_only_encoder, self).__init__()
        self.num_classes_OD = num_classes_OD
        self.num_anchors = 6
        self.nms_threshold = 0.4

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
        
        self.ctx_output1    = self._make_conv_block(128, 256, 1, 1, 0, 1, 1)
        self.ctx_output32   = self._make_conv_block(256, 256, 1, 1, 0, 1, 1)
        self.ctx_output32_ = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output2    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)
        self.ctx_output2_  = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output3    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)
        self.ctx_output3_  = self._make_conv_block(256, 256, 3, 1, 1, 1, 4)
        self.ctx_output4    = self._make_conv_block(512, 256, 1, 1, 0, 1, 1)

        self.regressionModel = RegressionModel(num_features_in = 256, num_anchors = 6)
        self.classificationModel = ClassificationModel(num_features_in = 256, num_anchors = 6, num_classes= self.num_classes_OD)

        self.poolind2d = nn.MaxPool2d(2,2)
        self.bilinear2d =nn.Upsample(scale_factor=2, mode='bilinear')

        self.anchors = Anchors_SSD()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()   
        self.output_act = nn.Sigmoid()
        self.focalLoss = losses.FocalLoss()
        # self.softmax = nn.Softmax(dim = 2)

    def _make_conv_block(self, in_channel, out_channel, kernel_size, stride, padding, dilation = 1, group = 1):
        layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation ,group),
                               nn.BatchNorm2d(out_channel),
                            nn.ReLU()
        )
        return layers

    def loc_permute(self, loc_tensor):
        loc_tensor = loc_tensor.permute(0, 2, 3, 1)
        return loc_tensor.contiguous().view(loc_tensor.shape[0], -1, 4)

    

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x = self.conv1a(img_batch)
        x = self.conv1b(x) #256
        x = self.poolind2d(x)
        x = self.res2a_branch2a(x) #128
        x = self.res2a_branch2b(x)
        x = self.poolind2d(x)
        x = self.res3a_branch2a(x) #64
        res3_out = self.res3a_branch2b(x)
        x = self.poolind2d(res3_out)
        x = self.res4a_branch2a(x) #32
        res4_out = self.res4a_branch2b(x)
        x = self.poolind2d(res4_out)
        x = self.res5a_branch2a(x) #16
        res5_out = self.res5a_branch2b(x)
        pool6 = self.poolind2d(res5_out)
        pool7 = self.poolind2d(pool6)

        ctx_output1   = self.ctx_output1(res3_out)
        ctx_output32  = self.ctx_output32(res4_out)
        ctx_output32_ = self.ctx_output32_(ctx_output32)
        ctx_output2   = self.ctx_output2(res5_out)
        ctx_output2_  = self.ctx_output2_(ctx_output2)
        ctx_output3  = self.ctx_output3(pool6)
        ctx_output3_ = self.ctx_output3_(ctx_output3)
        ctx_output4 = self.ctx_output4(pool7)

        regression = torch.cat([self.regressionModel(feature) for feature in [ctx_output1, ctx_output32, ctx_output32_, ctx_output2, ctx_output2_, ctx_output3, ctx_output3_, ctx_output4]], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in [ctx_output1, ctx_output32, ctx_output32_, ctx_output2, ctx_output2_, ctx_output3, ctx_output3_, ctx_output4]], dim=1)
        
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
