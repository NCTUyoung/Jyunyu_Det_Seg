from __future__ import print_function, division
import os
import sys

ROOT_DIR = os.path.realpath(__file__).split("demo/demo_ROS.py")[0]
sys.path.append(ROOT_DIR)

import threading
import numpy as np
import cv2

import math


import time
import importlib
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network.net import net_option
from utils.logger import MyLog
from utils.core_utils import count_parameters


import matplotlib.pyplot as  plt

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


from multiprocessing import  Queue

######################################
# Global Parameter
######################################
softmax = torch.nn.Softmax(dim = 1)
SHOW_SEG_RESULT = True
confidence_threshold = 0.7
DEBUG_MIN_OBJ_WIDTH = True
wmin = 100

pick_interval = 8
min_distance_threshold = 50
min_angle_threshold = 60



LINE_WIDTH = 2

color_polate_4cls = {1: "#00FF00",
                     2: "#0000FF",
                     3: "#FFFF00",
                     4: "#00FFFF",
                     5: "#FF0000"}

color_polate_4cls_QT = {3: "FF0",
                        4: "0FF",
                        5: "F00"}


cmap_4cls = {1: (  0,255,  0),
             2: (  0,  0,255),
             3: (255,255,  0),
             4: (  0,255,255),
             5: (255,  0,  0)}
alpha = 0.5


INPUT_SHAPE = (256,512 ,3) #row col



model_type = 'Jacinto_256x512_v3'
model_check_point = 'weights/detseg_ivs_revise_ssd_anchor/Jacinto_256x512_v3_256x512_detection_v1_ivs_bs_16_lr_1e-05_fixbackbone_True_freeze_bn_True_sampler_False_focaloss_True_revise_anchor_499.pt'
od_threshold = 0.3

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



######################################
# Global Parameter
######################################





######################################
# Function Util
######################################
def upsample_back(pred, size):
    """ 
    upsmaple back if segmentation pred if its size not equal to "size"
    Args:
       (1) pred: numpy array with size (H',W')
       (2) size: tuple (H,W)
    Return:
       (1) pred: numpy array with size (H',W')
    """
    if pred.shape != size:
        pred = cv2.resize(pred, (size[1], size[0]), 0, 0, interpolation = cv2.INTER_NEAREST)
        return pred
    else:
        return pred
def lane_post_process_find_mid(pred_seg, prob_map, pred_seg_max, img_draw):
    """Self define cluster method, try to find the mid points of each row """
    ###########################
    # get local max & cluster #
    ###########################
    tic = time.time()
    y_value = pred_seg.shape[1]-40
    # sample point contaion : (x, y, vector, label)
    sampled_points_info = []
    prob_map[:, 0] = 0.0 
    prob_map[:, -1] = 0.0 
    while(y_value > 0):
        #####################
        # get local maximum #
        #####################
        ticc = time.time()
        difference = np.diff(prob_map[y_value])
        difference[difference < 0] = -difference[difference < 0]
        difference[difference < confidence_threshold-0.1] = 0
        where = np.where(difference)
        indexes = where[0][1::2] + where[0][::2]
        indexes = indexes//2
        tocc = time.time()
        if indexes.shape == 0:
            y_value -= pick_interval
            continue
        if len(sampled_points_info) == 0:
            # for the first point, the angle can only be none
            # point: [x, y, angle, label]
            sampled_points_info = [[[point_x, y_value, None, pred_seg_max[y_value, point_x]]] for point_x in indexes]
            y_value -= pick_interval
            continue
        ####################
        # start clustering #
        ####################
        for point_x in indexes:
            # [distance, direction]
            new_info = [[point_distance((cluster[-1][0], cluster[-1][1]), (point_x, y_value)), \
                 point_degree(cluster[-1][2], (point_x - cluster[-1][0], y_value - cluster[-1][1]))] for cluster in sampled_points_info]
            new_info = np.array(new_info)
            min_distance_index = np.argmin(new_info[:, 0])
            min_distance = np.min(new_info[:, 0])
            angle = new_info[min_distance_index, 1]
            # print("angle min_distance", angle, min_distance)
            # if (min_distance < min_distance_threshold):
            if (min_distance < min_distance_threshold and angle is None) or\
               (min_distance < min_distance_threshold and angle < min_angle_threshold and angle is not None):
                # print("add to cluster")
                new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_distance_index][-1][0], sampled_points_info[min_distance_index][-1][1]))
                if sampled_points_info[min_distance_index][-1][2] is not None:
                    average_vector = ((new_unit_vector[0]+sampled_points_info[min_distance_index][-1][2][0])/2,
                                    (new_unit_vector[1]+sampled_points_info[min_distance_index][-1][2][1])/2 )
                    sampled_points_info[min_distance_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
                else:
                    sampled_points_info[min_distance_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            else:
                sampled_points_info.append([[point_x, y_value, None, pred_seg_max[y_value, point_x]]])
        y_value -= pick_interval
    toc = time.time()
    print("Clustering time: {}".format(toc - tic))
    ##########################
    # draw clustering result #
    ##########################
    tic = time.time()
    queue_info = [] 
    for i, cluster in enumerate(sampled_points_info):
        cluster = np.array(cluster)
        if cluster.shape[0] < 5:
            continue
        # vote using class label
        unique, count = np.unique(cluster[:, -1], return_counts = True)
        cluster_class = unique[np.argmax(count)]
        if cluster_class == 0:
            continue
        cluster = cluster[cluster[:,-1] == cluster_class]    
        ############
        # poly fit #
        ############
        order = 2
        cluster_xy = cluster[:,:2] 
        cluster_xy = cluster_xy.astype(np.int32)
        # judge if use poly fit or not
        if cluster_xy[:, 0][0] > cluster_xy[:, 0][-1]:
            cluster_xy = cluster_xy[::-1,:]
        x_cor_diff = np.diff(cluster_xy[:, 0])
        # if True:
        if np.where(x_cor_diff < 0)[0].shape[0] == 0:
            """case to use poly fit"""
            poly_parameter = np.polyfit(cluster_xy[:,0], cluster_xy[:, 1], order)
            poly = np.poly1d(poly_parameter)
            x, y = cluster_xy[:, 0], poly(cluster_xy[:, 0])

        else:
            poly_parameter = np.polyfit(cluster_xy[:,1], cluster_xy[:, 0], order)
            poly = np.poly1d(poly_parameter)
            # x, y = cluster_xy[:, 0], poly(cluster_xy[:, 0])
            x, y = poly(cluster_xy[:, 1]), cluster_xy[:, 1]

        queue_info.append([x, y, cluster_class])
        y_value -= pick_interval
    toc = time.time()
    print("DBSCAN time: {}".format(toc - tic))
    queue_info = [img_draw] + queue_info
    return queue_info

    
def point_distance(point1, point2):
    # print(point1, point2)
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
def point_degree(vector1, vector2):
    """
    Compute angle between two vector
    Args:
        (1) vector1: vector in the cluster
        (2) vector2: new vector obtained from new points and cluster point
    Return:
        (1) angle: angle between two vector
    """
    if vector1 == None:
        return None
    # cos = dot(v1,v2)/len(v1, v2)
    cos = (vector1[0] * vector2[0] + vector1[1] * vector2[1])/(((vector1[0]**2 + vector1[1]**2)**0.5)*((vector2[0]**2 + vector2[1]**2)**0.5))
    if cos >= 1:
        cos = 1.0
    if cos <= -1:
        cos = -1.0
    angle = 180*math.acos(cos)/3.14
    # print("angle", angle)
    return angle

def get_unit_vector(point1, point2):
    """get unit vector from point2 point to point1
    Args:
    (1) point1: end point 
    (2) point2: start point
    Return:
    (1) unit_vector: unit vector from point2 point to point1
    """
    vector_length = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
    return ((point1[0] - point2[0])/vector_length, (point1[1] - point2[1])/vector_length)


class Img_Sub():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/panaroma2",Image,self.update_image)
        self.img =  None
        self.image_lock = threading.RLock()
    def callback(self,data):
        
        self.q.put(img.copy())
    def update_image(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        img = cv2.resize(cv_image, (INPUT_SHAPE[1], INPUT_SHAPE[0]), 0, 0, interpolation = cv2.INTER_LINEAR)

        d = map(ord, img.data)
        arr = np.ndarray(shape=img,
                         dtype=np.int,
                         buffer=np.array(d))[:,:,::-1]

        if self.image_lock.acquire(True):
            self.img = arr
            self.image_lock.release()


def Segmentation_Det_task(img,net):
    RGB = np.zeros(INPUT_SHAPE,dtype=np.uint8)
    #resize image
    img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]), 0, 0, interpolation = cv2.INTER_LINEAR)
    #BRG to RGB
    img = img[:,:,[2,1,0]]
    # transfrom and expand dim
    input = img.transpose((2, 0, 1))
    input = np.expand_dims(input, axis = 0)
    # push to tensor
    input = torch.from_numpy(input).float()
    input = input.to(device)

    scores, classification, transformed_anchors, pred_seg_t = net(input)
    pred_seg = softmax(pred_seg_t)


    
    # threhold
    # pred_seg[pred_seg < confidence_threshold] = 0
    # back to cpu
    pred_seg = pred_seg.squeeze(dim = 0)
    pred_seg = pred_seg.data.cpu().numpy()
    pred_seg[pred_seg < confidence_threshold] = 0
    img_draw = img.astype(np.uint8)
    prob_map = np.max(pred_seg[3:, :, :], axis = 0)
    pred_seg_max = np.argmax(pred_seg, axis = 0)

    # upsample back if prediction size not equal to input_size
    pred_seg_max_up = upsample_back(pred_seg_max, INPUT_SHAPE)
    overlay_flag = np.zeros((INPUT_SHAPE[0], INPUT_SHAPE[1]))
    # draw
    for key, color in cmap_4cls.items():
        if key == 0:
            continue
        if key >= 3:
            continue
        else:
            RGB[pred_seg_max_up == key] = np.array(color)
            overlay_flag[pred_seg_max_up == key] = 1
    overlay = cv2.addWeighted(img_draw, alpha, RGB, (1-alpha), 0)
    img_draw[overlay_flag == 1] = overlay[overlay_flag == 1]
    img_draw_down = img_draw.copy()
    

    ##################
    # draw detection #
    ##################
    tic = time.time()
    scores = scores.data.cpu().numpy()
    idxs = np.where(scores > od_threshold) 
    for j in range(idxs[0].shape[0]):
        class_ =int(classification[idxs[0][j]])
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(round(bbox[0]))
        y1 = int(round(bbox[1]))
        x2 = int(round(bbox[2]))
        y2 = int(round(bbox[3]))
        # ped
        if class_ == 0 or class_ == 1:
            cv2.rectangle(img_draw_down, (x1,y1), (x2, y2), (0,255,0), LINE_WIDTH) 
        # car
        elif class_ == 2 or class_ == 3 or class_ == 4:
            cv2.rectangle(img_draw_down, (x1,y1), (x2, y2), (0,0,255), LINE_WIDTH) 
        # motor
        elif class_ == 5 or class_ == 6 or class_ == 7:
            cv2.rectangle(img_draw_down, (x1,y1), (x2, y2), (255,0,0),  LINE_WIDTH) 
        else:
            cv2.rectangle(img_draw_down, (x1,y1), (x2, y2), (0,0,0), LINE_WIDTH) 


    return img_draw_down,img_draw


def main():

    
    net = net_option(model_type, mode = "end2end")
    net = net.to(device)

    skip_frame = 1
    count = 0
    net.target_available = False
    net.eval()

    ######################################
    # Load Model
    ######################################

    # resume from checkpoint
    assert os.path.exists(model_check_point), "Checkpoint {} does not exist.".format(model_check_point)
    state = torch.load(model_check_point)
    net.load_state_dict(state["model_state"])
    bridge = CvBridge()
    image_sub_node = Img_Sub()
    image_pub_seg = rospy.Publisher("Jyuntu/seg",Image)
    image_pub_seg_det = rospy.Publisher("Jyuntu/seg_det",Image)

    rospy.init_node('Jyunyu_Det_seg', anonymous=True)

    while not rospy.is_shutdown():
        try:
            img = image_sub_node.img
            seg_det,seg = Segmentation_Det_task(img,net)
            image_pub_seg.publish(bridge.cv2_to_imgmsg(seg, "bgr8"))
            image_pub_seg_det.publish(bridge.cv2_to_imgmsg(seg_det, "bgr8"))
        except KeyboardInterrupt:
            os.system('pkill -9 python')
            print("Shutting down")

        
if __name__ == '__main__':
    main()    

