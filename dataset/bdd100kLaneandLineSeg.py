#######################################
#
#
#
#######################################
from __future__ import print_function, division
import os
import cv2
import numpy as np
import torch
import multiprocessing
from utils.labels_lane import labels_lane_line 
cv2.setNumThreads(0)

######################################
# for Sofrmax cross entropy training #
######################################
# 0: background 1~4: lanes 
class bdd100kLandandLineDataset(object):
    def __init__(self, input_size, split = "train", augmentation = None, label = "reduce_v3"):
        self.input_size = input_size
        self.split = split
        self.augmentation = augmentation
        # self.label_type = label
        # assert self.label_type in ["all", "reduce", "reduce_v2", "reduce_v3"]
        # print("label_type", self.label_type)

        # self._dataset_path = os.path.join("..", "datasets", "bdd100k", "bdd100k")
        # self._image_path = os.path.join(self._dataset_path, "images", "100k", split)
        # self._segmap_path = os.path.join(self._dataset_path, "lane_line_seg", "color_labels", split)
        # # self._list_file = os.path.join(self._dataset_path, "lane_seg_list", "100k", split, "list.txt")
        # self._list_file = os.path.join(self._dataset_path, "lane_seg_list", "100k", split, "list_wo_backgroud.txt")
        # self._data_list = [line.rstrip("\n") for line in open(self._list_file, "r")]

        self._num_classes = 6

    def encode_segmap(self, seg):
        seg_mask = np.zeros(seg.shape[0:2])    
        for label in labels_lane_line:
            # print(label.color)
            seg_mask[(np.sum(seg == np.array(label.color)[::-1], axis = 2)) == 3] = label.trainId
        return seg_mask

    def __getitem__(self, index):
        # while True:
        #     img_path = os.path.join(self._image_path, self._data_list[index]+".jpg")
        #     seg_path = os.path.join(self._segmap_path, self._data_list[index]+"_seg.png")
        #     # read segmentation map
        #     seg_color = cv2.imread(seg_path) 
        #     if np.max(seg_color) == 0 :
        #         # index += 1
        #         break
        #     else:
        #         break
        # self._data_list[index] = "428f2b79-40cab628"
        img_path = os.path.join(self._image_path, self._data_list[index]+".jpg")
        seg_path = os.path.join(self._segmap_path, self._data_list[index]+"_lane_line_color.png")
        seg_color = cv2.imread(seg_path) 
        # print(seg_path)
        seg_color = seg_color[:,:, [2,1,0]]
        # seg_color = seg_color.astype(np.uint8) 
        # seg_mask[seg_mask == 4] = 255
        # cv2.imshow("seg", seg_color)
        # cv2.waitKey(0)
        # encode segmentation map
        seg_mask = self.encode_segmap(seg_color)
        # read image
        img = cv2.imread(img_path)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        img = img.astype(np.float32)
        # bgr to rgb
        img = img[:,:, [2,1,0]]
        # resize 
        # print(seg_mask.shape)
        # print(np.unique(seg_mask))
        # print(np.max(seg_mask))
        # seg_mask = seg_mask.astype(np.uint8) 
        # seg_mask[seg_mask == 1] = 255
        # cv2.imshow("seg", seg_mask)
        # cv2.waitKey(0)

        # augmentation if needed
        if self.augmentation:
            img, seg_mask = self.augmentation(img, seg_mask)
        # dim transformation
        img = img.transpose((2, 0, 1))

        # seg_mask = seg_mask.astype(np.uint8) 
        # seg_mask[seg_mask == 4] = 255
        # print(np.max(seg_mask))
        # cv2.imshow("seg", seg_mask)
        # cv2.waitKey(0)

        return torch.from_numpy(img).float(), torch.from_numpy(seg_mask).long()

    def __len__(self):
        return len(self._data_list)

    @property
    def num_classes(self):
        return self._num_classes
    @property 
    def color_map(self):
        _cmap = { # in RGB order
          0: (128, 64,128),
          1: (240,248,255),
          2: (  0,  0,255),
          3: (138, 43,226),
          4: (255,255,  0),
          5: (173,255, 47),
          6: (154,205, 50),
          7: ( 30,144,255),
          8: (165, 42, 42)}
        return _cmap

    @property
    def dataset_path(self):
        return self._dataset_path
    
    @property
    def data_list(self):
        return self._data_list
    