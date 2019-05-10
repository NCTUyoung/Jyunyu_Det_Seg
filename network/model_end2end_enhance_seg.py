from __future__ import print_function
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from network.net_utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from network.anchors import Anchors
import network.losses as losses #focal loss for detection
from network.loss_seg import CrossEntropyLoss2d 
from lib.nms.pth_nms import pth_nms

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

pretrain_store_dir = os.path.join("weights", "pretrain")

def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    return pth_nms(dets, thresh)

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        ####################
        # detection subnet #
        ####################
        # upsample C5 to get P5 from the FPN paper
        self.P5_conv          = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_top           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4_conv          = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_top           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.P3_conv = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_top = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6_conv = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_conv = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        #######################
        # segmentation subnet #
        #######################
        self.K5_conv_1x1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride = 1, padding = 0)
        self.K4_conv_1x1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride = 1, padding = 0)
        self.K3_conv_1x1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride = 1, padding = 0)
        self.K2_conv_1x1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride = 1, padding = 0)
        # after pyramid, smooth feaure map by conv at the top
        self.K5_top = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)
        self.K4_top = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)
        self.K3_top = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)
        self.K2_top = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)
    def upsmaple_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: top feature map to be upsampled.
          y: lateral feature map.

        Returns:
          added feature map.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='nearest', align_corners=None) + y  # bilinear, False

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs # C3 size :(1, 512, 64, 128) , C4 size :(1, 1024, 32, 64), C5 size :(1, 2048, 16, 32) 
        ####################
        # detection subnet #
        ####################
        p5 = self.P5_conv(C5)
        p4 = self.upsmaple_add(p5, self.P4_conv(C4))
        p3 = self.upsmaple_add(p4, self.P3_conv(C3))

        # after pyramid, do conv on the top
        p3 = self.P3_top(p3)
        p4 = self.P4_top(p4)
        p5 = self.P5_top(p5)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        p6 = self.P6_conv(C5)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        p7 = self.P7_conv(F.relu(p6))

        #######################
        # segmentation subnet #
        #######################
        k5 = self.K5_conv_1x1(C5)
        k4 = self.upsmaple_add(k5, self.K4_conv_1x1(C4))
        k3 = self.upsmaple_add(k4, self.K3_conv_1x1(C3))
        k2 = self.upsmaple_add(k3, self.K2_conv_1x1(C2))
        # after pyramid, do conv on the top
        k5 = self.K5_top(k5)
        k4 = self.K4_top(k4)
        k3 = self.K3_top(k3)
        k2 = self.K2_top(k2)
        # print(self.K2_top.weight)
        return [[k2, k3, k4, k5],[p3, p4, p5, p6, p7]]


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

class ResNet(nn.Module):

    def __init__(self, num_classes_seg, num_classes_OD, block, layers, mode = "detection"):
        super(ResNet, self).__init__()
        self.target_available = True
        self.nms_threshold = 0.4
        self.mode = mode
        print("nms_threshold", self.nms_threshold)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #
        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        ########################
        # segmentation subnet  #
        ########################
        """Enhance version with dilated rate = 4"""
        self.D2_conv_3x3_1 = nn.Conv2d(256, 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D2_conv_3x3_2 = nn.Conv2d(64 , 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D3_conv_3x3_1 = nn.Conv2d(256, 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D3_conv_3x3_2 = nn.Conv2d(64 , 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D4_conv_3x3_1 = nn.Conv2d(256, 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D4_conv_3x3_2 = nn.Conv2d(64 , 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D5_conv_3x3_1 = nn.Conv2d(256, 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.D5_conv_3x3_2 = nn.Conv2d(64 , 64, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.upsample1 = nn.Upsample(scale_factor=8, mode='nearest', align_corners=None)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest', align_corners=None)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv_final_smooth = nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1)
        """ extran convolution for upsample  """
        self.conv_up1 = nn.Conv2d(128,128,kernel_size = 3, stride = 1, padding = 1) 
        self.conv_up2 = nn.Conv2d(128,128,kernel_size = 3, stride = 1, padding = 1) 
        self.conv_final = nn.Conv2d(128, num_classes_seg, kernel_size = 1, stride = 1, padding = 0) # num_classes = 19
        self.seg_loss = CrossEntropyLoss2d()
        ####################
        # detection subnet #
        ####################
        # self.regressionModel = RegressionModel(256, num_anchors = 12)
        # self.classificationModel = ClassificationModel(256, num_anchors = 12, num_classes=num_classes_OD)
        self.regressionModel = RegressionModel(256, num_anchors = 15)
        self.classificationModel = ClassificationModel(256, num_anchors = 15, num_classes=num_classes_OD)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()   
        self.focalLoss = losses.FocalLoss()
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()
        self.freeze_bn2()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def freeze_bn2(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                for para in layer.parameters():
                    para.requires_grad = False 

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

    def forward(self, inputs):
        if self.mode == "detection":
            return self.forward_detection(inputs)
        elif self.mode == "segmentation":
            return self.forward_segmentation(inputs)
        else: #end2end
            if self.target_available:
                # img_batch, target = inputs
                img_batch, annot_det, annot_seg = inputs
            else:
                img_batch = inputs
            x = self.conv1(img_batch)
            x = self.bn1(x)
            # print(self.bn1.weight)
            x = self.relu(x)
            c1 = self.maxpool(x)

            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            # print(c2.size(), c3.size(), c4.size(), c5.size())
            [features_segmentation, features_detection] = self.fpn([c2 ,c3, c4, c5])
            (k2, k3, k4, k5) = (features_segmentation[0], features_segmentation[1], features_segmentation[2], features_segmentation[3])
            ##############################
            # segmentaion subnet forward #
            ##############################
            d2 = F.relu(self.D2_conv_3x3_1(k2))
            d2 = F.relu(self.D2_conv_3x3_2(d2))
            d3 = F.relu(self.D3_conv_3x3_1(k3))
            d3 = F.relu(self.D3_conv_3x3_2(d3))
            d4 = F.relu(self.D4_conv_3x3_1(k4))
            d4 = F.relu(self.D4_conv_3x3_2(d4))
            d5 = F.relu(self.D5_conv_3x3_1(k5))
            d5 = F.relu(self.D5_conv_3x3_2(d5))
            # print("d2.size(),,d3.size(), d4.size(), d5.size()",d2.size(),d3.size(), d4.size(), d5.size())
            d3 = self.upsample3(d3)
            d4 = self.upsample2(d4)
            d5 = self.upsample1(d5)

            # d_final = self.concat(d2, d3, d4, d5)
            d_final = torch.cat((d2, d3, d4, d5), 1)
            d_smooth = F.relu(self.conv_final_smooth(d_final))
            ##################################
            # extra conv to upsmaple back
            ##################################
            d_up1 = self.upsample3(d_smooth)
            d_up1 = F.relu(self.conv_up1(d_up1))
            d_up2 = self.upsample3(d_up1)
            d_up2 = F.relu(self.conv_up2(d_up2))
            pred_seg = self.conv_final(d_up2)
            # if self.target_available:
            #     pred_seg = self.upsample_back(target, pred_seg)
            #     seg_loss = self.seg_loss(pred_seg, target)
                # return seg_loss, pred_seg
            # else:
                    # return pred_seg
            #############################
            # detection subnet forward  #
            #############################
            regression = torch.cat([self.regressionModel(feature) for feature in features_detection], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in features_detection], dim=1)
            anchors = self.anchors(img_batch)

            if self.training:
                # detection
                focal_loss = self.focalLoss(classification, regression, anchors, annot_det)
                # segmentation
                pred_seg = self.upsample_back(annot_seg, pred_seg)
                seg_loss = self.seg_loss(pred_seg, annot_seg)
                return [focal_loss, seg_loss, pred_seg]
            elif not self.training and self.target_available:
                # detection
                # focal_loss =  self.focalLoss(classification, regression, anchors, annot_det)
                # transformed_anchors = self.regressBoxes(anchors, regression)
                # transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                # scores = torch.max(classification, dim=2, keepdim=True)[0]

                # scores_over_thresh = (scores>0.05)[0, :, 0]

                # if scores_over_thresh.sum() == 0:
                #     # no boxes to NMS, just return
                #     return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), pred_seg, seg_loss]

                # classification = classification[:, scores_over_thresh, :]
                # transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                # scores = scores[:, scores_over_thresh, :]

                # anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

                # nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
                # # segmentation
                # pred_seg = self.upsample_back(annot_seg, pred_seg)
                # seg_loss = self.seg_loss(pred_seg, annot_seg)
                # return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg, seg_loss]
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
                scores_over_thresh = (scores>0.06)[0, :, 0]
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just return
                    return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), pred_seg]

                classification = classification[:, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                scores = scores[:, scores_over_thresh, :]

                anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], self.nms_threshold)

                nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
                # print(transformed_anchors[0, anchors_nms_idx, :])
                # print(transformed_anchors[0, anchors_nms_idx, :].size())
                # segmentation
                pred_seg = pred_seg #8x downsample
                return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], pred_seg]

            
    def forward_segmentation(self, inputs):
        if self.target_available:
            img_batch, target = inputs
        else:
            img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        # print(self.bn1.weight)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # print(c2.size(), c3.size(), c4.size(), c5.size())
        [features_segmentation, features_detection] = self.fpn([c2 ,c3, c4, c5])
        (k2, k3, k4, k5) = (features_segmentation[0], features_segmentation[1], features_segmentation[2], features_segmentation[3])
        ##############################
        # segmentaion subnet forward #
        ##############################
        d2 = F.relu(self.D2_conv_3x3_1(k2))
        d2 = F.relu(self.D2_conv_3x3_2(d2))
        d3 = F.relu(self.D3_conv_3x3_1(k3))
        d3 = F.relu(self.D3_conv_3x3_2(d3))
        d4 = F.relu(self.D4_conv_3x3_1(k4))
        d4 = F.relu(self.D4_conv_3x3_2(d4))
        d5 = F.relu(self.D5_conv_3x3_1(k5))
        d5 = F.relu(self.D5_conv_3x3_2(d5))
        # print("d3.size(), d4.size(), d5.size()",d3.size(), d4.size(), d5.size())
        d3 = self.upsample3(d3)
        d4 = self.upsample2(d4)
        d5 = self.upsample1(d5)

        # d_final = self.concat(d2, d3, d4, d5)
        d_final = torch.cat((d2, d3, d4, d5), 1)
        d_smooth = F.relu(self.conv_final_smooth(d_final))
        ##################################
        # extra conv to upsmaple back
        ##################################
        d_up1 = self.upsample3(d_smooth)
        d_up1 = F.relu(self.conv_up1(d_up1))
        d_up2 = self.upsample3(d_up1)
        d_up2 = F.relu(self.conv_up2(d_up2))
        pred_seg = self.conv_final(d_up2)
        # if self.target_available:
        #     pred_seg = self.upsample_back(target, pred_seg)
        #     seg_loss = self.seg_loss(pred_seg, target)
            # return seg_loss, pred_seg
        # else:
                # return pred_seg

        # print(pred_seg.size())
        if self.target_available:
            pred_seg = self.upsample_back(target, pred_seg)
            loss = self.seg_loss(pred_seg, target)
            # print(loss)
            # x = input("pause")
            return loss, pred_seg
        else:
            return pred_seg

    def forward_detection(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            
        x = self.conv1(img_batch)
        x = self.bn1(x)
        # print(self.bn1.weight)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # print(c2.size(), c3.size(), c4.size(), c5.size())
        [features_segmentation, features_detection] = self.fpn([c2 ,c3, c4, c5])
 
        #############################
        # detection subnet forward  #
        #############################
        regression = torch.cat([self.regressionModel(feature) for feature in features_detection], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features_detection], dim=1)
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

def resnet18(num_classes_seg, num_classes_OD, pretrained=False, mode = "detection", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes_seg, num_classes_OD, BasicBlock, [2, 2, 2, 2], mode,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=pretrain_store_dir), strict=False)
    return model


def resnet34(num_classes_seg, num_classes_OD, pretrained=False, mode = "detection", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes_seg, num_classes_OD, BasicBlock, [3, 4, 6, 3], mode, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=pretrain_store_dir), strict=False)
    return model


def resnet50(num_classes_seg, num_classes_OD, pretrained=False, mode = "detection", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes_seg, num_classes_OD, Bottleneck, [3, 4, 6, 3], mode, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=pretrain_store_dir), strict=False)
    return model

def resnet101(num_classes_seg, num_classes_OD, pretrained=False, mode = "detection", **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes_seg, num_classes_OD, Bottleneck, [3, 4, 23, 3], mode, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=pretrain_store_dir), strict=False)
    return model


def resnet152(num_classes_seg, num_classes_OD, pretrained=False, mode = "detection", **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes_seg, num_classes_OD, Bottleneck, [3, 8, 36, 3], mode, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir=pretrain_store_dir), strict=False)
    return model