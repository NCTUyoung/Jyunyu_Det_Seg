import torch
import torch.nn as nn
import numpy as np

BOX_REG_V2 = False


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=64):
    # def __init__(self, num_features_in, num_anchors=9, feature_size=256):
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
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=64):
    # def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None, normalize_coor = False):
        super(BBoxTransform, self).__init__()
        self.normalize_coor = normalize_coor
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None and not BOX_REG_V2:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        elif std is None and BOX_REG_V2:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()    
        else:
            self.std = std

    def forward(self, boxes, deltas):
        if not BOX_REG_V2:
            # if not self.normalize_coor:
            if True:
                # print(boxes)
                # print(deltas)
                widths  = boxes[:, :, 2] - boxes[:, :, 0]
                heights = boxes[:, :, 3] - boxes[:, :, 1]
                ctr_x   = boxes[:, :, 0] + 0.5 * widths
                ctr_y   = boxes[:, :, 1] + 0.5 * heights

                dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
                dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
                dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
                dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

                pred_ctr_x = ctr_x + dx * widths
                pred_ctr_y = ctr_y + dy * heights
                pred_w     = torch.exp(dw) * widths
                pred_h     = torch.exp(dh) * heights
                # print("boxes", boxes)
                # print("deltas", deltas)
                pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
                pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
                pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
                pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

                if self.normalize_coor:
                    pred_boxes_x1 = pred_boxes_x1*512
                    pred_boxes_x2 = pred_boxes_x2*512
                    pred_boxes_y1 = pred_boxes_y1*256
                    pred_boxes_y2 = pred_boxes_y2*256

                pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
                # print(pred_boxes)
                return pred_boxes
            # else:
            #     widths  = boxes[:, :, 2] - boxes[:, :, 0]
            #     heights = boxes[:, :, 3] - boxes[:, :, 1]
            #     ctr_x   = boxes[:, :, 0] + 0.5 * widths
            #     ctr_y   = boxes[:, :, 1] + 0.5 * heights

            #     dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
            #     dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
            #     dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
            #     dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

            #     pred_ctr_x = ctr_x + dx * widths
            #     pred_ctr_y = ctr_y + dy * heights
            #     pred_w     = torch.exp(dw) * widths
            #     pred_h     = torch.exp(dh) * heights


        else:
            widths  = boxes[:, :, 2] - boxes[:, :, 0]
            heights = boxes[:, :, 3] - boxes[:, :, 1]

            dx1 = deltas[:, :, 0] * self.std[0] + self.mean[0]
            dy1 = deltas[:, :, 1] * self.std[1] + self.mean[1]
            dx2 = deltas[:, :, 2] * self.std[2] + self.mean[2]
            dy2 = deltas[:, :, 3] * self.std[3] + self.mean[3]


            pred_boxes_x1 = widths * dx1 + boxes[:, :, 0]
            pred_boxes_y1 = heights * dy1 * boxes[:, :, 1]
            pred_boxes_x2 = widths * dx2 + boxes[:, :, 2]
            pred_boxes_y2 = heights * dy1 * boxes[:, :, 3]
            pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
            return pred_boxes


# class BBoxTransform_SSD(nn.Module):

#     def __init__(self, mean=None, std=None):
#         super(BBoxTransform_SSD, self).__init__()
#         if mean is None:
#             self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
#         else:
#             self.mean = mean
#         if std is None and not BOX_REG_V2:
#             self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
#         elif std is None and BOX_REG_V2:
#             self.std = torch.from_numpy(np.array([0.2, 0.2, 0.2, 0.2]).astype(np.float32)).cuda()    
#         else:
#             self.std = std

#     def forward(self, boxes, deltas):
#         if not BOX_REG_V2:
#             widths  = boxes[:, :, 2] - boxes[:, :, 0]
#             heights = boxes[:, :, 3] - boxes[:, :, 1]
#             ctr_x   = boxes[:, :, 0] + 0.5 * widths
#             ctr_y   = boxes[:, :, 1] + 0.5 * heights

#             dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
#             dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
#             dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
#             dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

#             pred_ctr_x = ctr_x + dx * widths
#             pred_ctr_y = ctr_y + dy * heights
#             pred_w     = torch.exp(dw) * widths
#             pred_h     = torch.exp(dh) * heights

#             pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
#             pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
#             pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
#             pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

#             pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

#             return pred_boxes
#         else:
#             widths  = boxes[:, :, 2] - boxes[:, :, 0]
#             heights = boxes[:, :, 3] - boxes[:, :, 1]

#             dx1 = deltas[:, :, 0] * self.std[0] + self.mean[0]
#             dy1 = deltas[:, :, 1] * self.std[1] + self.mean[1]
#             dx2 = deltas[:, :, 2] * self.std[2] + self.mean[2]
#             dy2 = deltas[:, :, 3] * self.std[3] + self.mean[3]

#             pred_boxes_x1 = widths * dx1 + boxes[:, :, 0]
#             pred_boxes_y1 = heights * dy1 * boxes[:, :, 1]
#             pred_boxes_x2 = widths * dx2 + boxes[:, :, 2]
#             pred_boxes_y2 = heights * dy1 * boxes[:, :, 3]
#             pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
#             return pred_boxes

class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes


#################################
# FOR SSD TRAINING
#################################
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
    
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    # print(inter / union)
    # print("======")
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print(priors)
    overlaps = jaccard(
        truths,
        priors
        # point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, center_size(priors), variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    eps = 1e-6
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh+eps) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
