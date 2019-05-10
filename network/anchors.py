import numpy as np
import torch
import torch.nn as nn
import cv2

DEBUG_ANCHOR = False
class Anchors(nn.Module):
    def __init__(self, input_shape = None,pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        self.input_shape = input_shape
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            # self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0), 2 ** (-1.0 / 3.0)])
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0), 2 ** (-1.0 / 3.0), 2 ** (-3.0 / 3.0)])
        ###################
        # if fix size set #
        ###################
        # if self.input_shape is not None:
        #     self.image_shape = np.array(self.input_shape)
    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        #############################
        # if not fix size image set #
        #############################
        # if self.input_shape is None:
        #     image_shape = image.shape[2:]
        #     image_shape = np.array(image_shape)
        # else:
        #     image_shape = self.image_shape
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            if idx == 0 and DEBUG_ANCHOR:
                self.visualize_anchor(shifted_anchors, image)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        # self.visualize_anchor(all_anchors)
        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()

    def visualize_anchor(self, anchors, image):
        print("visualize_anchor")
        image_draw = image.data.cpu().numpy().astype(np.uint8)
        image_draw = image_draw[0]
        image_draw = image_draw.transpose((1,2,0))
        image_draw_copy = image_draw.copy()
        print(image_draw.shape)
        print(type(image_draw))
        print(image_draw.dtype)
        anchors[anchors<0] = 0
        for index, anchor in enumerate(anchors):
            if index < int(anchors.shape[0]/20):
                continue
            # if not index % 2 == 0:
            #     continue
            print(int(anchor[0]), int(anchor[1]),int(anchor[2]),int(anchor[3]))
            print("width {}, height {}".format(anchor[2]-anchor[0],anchor[3]-anchor[1]))
            cv2.rectangle(image_draw_copy, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])),color = (0,0,255))
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image_draw_copy)
            cv2.waitKey(0)
def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios x scales w.r.t. a reference window.
    For one grid.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors to store x y w h and then transform to x1 y1 x2 y2
    anchors = np.zeros((num_anchors, 4))

    # scale base_size # w and h
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    # output dim = (num_anchors, 1)
    areas = anchors[:, 2] * anchors[:, 3]
    # print("areas.shape", areas.shape)
    # print("np.shape(np.repeat(ratios, len(scales)))", np.shape(np.repeat(ratios, len(scales))))
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

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


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

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

