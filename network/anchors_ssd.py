import numpy as np
import torch
import torch.nn as nn
import cv2

DEBUG_ANCHOR = False
class Anchors_SSD(nn.Module):
    def __init__(self, input_shape = None, pyramid_levels=None, min_sizes=None, max_sizes=None, offsets = None, ratios=None, scales=None,  normalize_anchor = False):
        super(Anchors_SSD, self).__init__()
        self.input_shape = input_shape
        self.pyramid_levels = pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.offsets = offsets
        self.ratios = np.array(ratios)
        self.scales = np.array([1])
        self.normalize_anchor = normalize_anchor
        print("self.pyramid_levels", self.pyramid_levels)
        print("self.min_sizes", self.min_sizes)
        print("self.max_sizes", self.max_sizes)
        print("self.offsets", self.offsets)
        print("self.normalize_anchor", self.normalize_anchor)
    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        # SSD
        image_shapes = [(image_shape) // (2 ** x) for x in self.pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        # print(self.pyramid_levels)
        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.min_sizes[idx], max_size=self.max_sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors, self.offsets[idx])
            if idx == 7 and DEBUG_ANCHOR:
            # if True:
                print("image_shapes[idx]", image_shapes[idx])
                self.visualize_anchor(shifted_anchors, image)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
        # print(all_anchors)
        # print("=================")
        if self.normalize_anchor:
            # print(image_shape)
            all_anchors[:,0::2] = all_anchors[:,0::2]/image_shape[1]
            all_anchors[:,1::2] = all_anchors[:,1::2]/image_shape[0]
            """ Clip normalize anchor"""          
            # all_anchors[all_anchors < 0.0] = 0.0
            # all_anchors[all_anchors > 1.0] = 1.0
        all_anchors = np.expand_dims(all_anchors, axis=0)
        # print("all_anchors", all_anchors)
        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()

    def visualize_anchor(self, anchors, image):
        print("visualize_anchor")
        image_draw = image.data.cpu().numpy().astype(np.uint8)
        image_draw = image_draw[0]
        image_draw = image_draw.transpose((1,2,0))
        image_draw_copy = image_draw.copy()
        # print(image_draw.shape)
        # print(type(image_draw))
        # print(image_draw.dtype)
        anchors[anchors<0] = 0
        # print(len(anchors))
        for index, anchor in enumerate(anchors):
            if index < int(anchors.shape[0]/20):
                continue
            # if not index % 2 == 0:
            #     continue
            print("x1, y1, x2, y2", int(anchor[0]), int(anchor[1]),int(anchor[2]),int(anchor[3]))
            print("width {}, height {}".format(anchor[2]-anchor[0],anchor[3]-anchor[1]))
            cv2.rectangle(image_draw_copy, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])),color = (0,0,255))
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image_draw_copy)
            cv2.waitKey(0)
            # break

def generate_anchors(base_size, max_size, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios x scales w.r.t. a reference window.
    For one grid.
    """

    # if ratios is None:
    #     ratios = np.array([0.5, 1, 2])

    # if scales is None:
    #     scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors to store x y w h and then transform to x1 y1 x2 y2
    anchors = np.zeros((num_anchors, 4)) 

    # scale base_size # w and h
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    """SSD BIG rectangel"""
    big_rec_size = (max_size * base_size)**0.5
    # anchors[1, 2:] = max_size * np.array([1, 1])
    anchors[1, 2:] = big_rec_size * np.array([1, 1])

    # compute areas of anchors
    # output dim = (num_anchors, 1)
    areas = anchors[2:, 2] * anchors[2:, 3]
    # print("areas.shape", areas.shape)
    # print("np.shape(np.repeat(ratios, len(scales)))", np.shape(np.repeat(ratios, len(scales))))
    # correct for ratios
    anchors[2:, 2] = np.sqrt(areas / np.repeat(ratios[2:], len(scales)))
    anchors[2:, 3] = anchors[2:, 2] * np.repeat(ratios[2:], len(scales))
    # print(anchors)
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # print(anchors)
    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


# def anchors_for_shape(
#     image_shape,
#     pyramid_levels=None,
#     ratios=None,
#     scales=None,
#     strides=None,
#     sizes=None,
#     shapes_callback=None,
# ):

#     image_shapes = compute_shape(image_shape, pyramid_levels)

#     # compute anchors over all pyramid levels
#     all_anchors = np.zeros((0, 4))
#     for idx, p in enumerate(pyramid_levels):
#         anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
#         shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
#         all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

#     return all_anchors


def shift(shape, stride, anchors, shift = 0.5):
    shift_x = (np.arange(0, shape[1]) + shift) * stride
    shift_y = (np.arange(0, shape[0]) + shift) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

