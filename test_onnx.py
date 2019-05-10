from __future__ import print_function, division
import onnx
import onnx_tensorrt.backend as backend
# import numpy as np

# model = onnx.load("test.onnx")
# engine = backend.prepare(model, device='CUDA:0')
# input_data = np.random.random(size=(32, 3, 512, 1024)).astype(np.float32)
# output_data = engine.run(input_data)[0]
# print(output_data)
# print(output_data.shape)

#############################################################
# Script for lane and line segmentation demo.
# Model was trained on BDD100K and test on bdd validation set.
# Show the lane segmentation result and post-process results
# Date: 20190412
#############################################################

import os
import sys
# ROOT_DIR = os.path.realpath(__file__).split("demo/test_onnx.py")[0]
# sys.path.append(ROOT_DIR)
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import interp1d
import scipy.signal as signal
# from sklearn.cluster import DBSCAN
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
from dataset.augmentation_seg import bdd100kLandSegAug

DEFINE_ONNX = True

softmax = torch.nn.Softmax(dim = 1)
SHOW_SEG_RESULT = True
confidence_threshold = 0.7

pick_interval = 8
min_distance_threshold = 50
# pick_interval = 1
# min_distance_threshold = 6
numbef_of_split = 20

color_polate_4cls = {1: "#00FF00",
                     2: "#0000FF",
                     3: "#FFFF00",
                     4: "#00FFFF",
                     5: "#FF0000"}
cmap_4cls = {1: (  0,255,  0),
             2: (255,  0,  0),
             3: (  0,255,255),
             4: (255,255,  0),
             5: (  0,  0,255)}

class drawer(object):
    def __init__(self):
        plt.connect('key_release_event', self.keyin_continue)
        # plt.ion()
        w, h = 16, 9
        self.fig = plt.figure(figsize = (w, h))
        #left bottom width height
        self.ax  = self.fig.add_axes([0.05, 0.50, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
        self.ax2 = self.fig.add_axes([0.55, 0.50, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
        self.ax3 = self.fig.add_axes([0.05, 0.05, 0.4, 0.4], frameon = True)
        self.ax4 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4], frameon = True)

        # self.ax2 = self.fig.add_axes([0.55, 0.50, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
        # self.ax3 = self.fig.add_axes([0.05, 0.05, 0.4, 0.4], frameon = True)
        # self.ax4 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4], frameon = True)
        # self.ax  = self.fig.add_axes([0.00, 0.00, 1.0, 1.0], frameon = True)  # If False, suppress drawing the figure frame.
        # self.ax.set_title("Segmentation Result")
        # self.ax2.set_title("Local Maximum")
        # self.ax3.set_title("Clustering")
        # self.ax4.set_title("Polyfit")

        # self.ax  = self.fig.add_axes([0.05, 0.05, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
        # self.ax2 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
        # self.ax3 = self.fig.add_axes([0.05, 0.50, 0.4, 0.4], frameon = True)
        # self.ax4 = self.fig.add_axes([0.55, 0.50, 0.4, 0.4], frameon = True)
        # self.custon_handle = custom_lines = [matplotlib.lines.Line2D([0], [0], color="#FF0000", lw=4),
        #         matplotlib.lines.Line2D([0], [0], color="#0000FF", lw=4),
        #         matplotlib.lines.Line2D([0], [0], color="#FFFF00", lw=4)]
        # self.fig.legend(self.custon_handle,
        # labels=('Single Line', 'Dashed Line', 'Double Line'),
        # loc='upper right')
    def keyin_continue(self, event):
        if event.key == 'n':
            plt.cla()
        else:
            return


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest = "model", type = str, help = "Network archtecture.")
    parser.add_argument("-i", dest = "input_size", type = int, nargs = "+", default = [512,1024], help = "Network input size.")
    parser.add_argument("-c", dest = "checkpoint", type = str, help = "Checkpoint file.")
    # parser.add_argument("-iv", dest = "input_video", type = str, required = True, help = "Input video for demo.")
    parser.add_argument("-ov", dest = "output_video", action = "store_true", help = "Input video for demo.")
    return parser.parse_args()

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

def lane_post_process(pred_seg, binary_map, prob_map, pred_seg_max, img_draw, draw, writer):
    draw.ax2.imshow(img_draw)
    draw.ax3.imshow(img_draw)
    draw.ax4.imshow(img_draw)
    # data = prob_map[350]
    # print(data)
    # draw.ax.plot(data)
    ##########
    # filter #
    ##########
    # window = signal.general_gaussian(51, p=0.5, sig=20)
    # filtered = signal.fftconvolve(window, data)
    # # draw.ax3.plot(filtered)
    # print(filtered)
    # print(filtered.shape)
    # filtered = (np.average(data) / np.average(filtered)) * filtered
    # filtered = np.roll(filtered, -25)
    # draw.ax3.plot(filtered)
    # # print(filtered)
    # # x = input("pause")
    # data = filtered
    # plt.draw()
    # plt.waitforbuttonpress()
    #################
    # get local max #
    #################
    # tic = time.time()
    # y_value = pred_seg.shape[1]-1
    # sampled_points_info = []
    # while(y_value > 0):
    #     indexes = signal.argrelextrema(prob_map[y_value], comparator=np.greater, order=2)
    #     if len(indexes[0]) == 0:
    #         y_value -= pick_interval
    #         continue
    #     corresponding_prob = [prob_map[y_value][point] for point  in indexes[0]]
    #     local_max_points = [[point, y_value, pred_seg_max[y_value, point]] for point in indexes[0]]
    #     sampled_points_info.extend(local_max_points)


    #     local_max_points = np.array(local_max_points)
    #     if local_max_points.shape[0] != 0:
    #         draw.ax.scatter(local_max_points[:,0], local_max_points[:, 1], color = "r")
    #     y_value -= pick_interval
    # toc =time.time()
    # print("Get global max time: {}".format(toc - tic))
    #######
    # IPM #
    #######
    ##########
    # DBSCAN #
    ##########
    # tic = time.time()
    # sampled_points_info = np.array(sampled_points_info)
    # if sampled_points_info.size == 0:
    #     return 0
    # sampled_points_xy = sampled_points_info[:, 0:2] 
    # sampled_points_label = sampled_points_info[:, 2] 
    # clustering = DBSCAN(eps=8, min_samples=5).fit(sampled_points_xy)
    # labels = clustering.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # unique_labels = set(labels)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 0.5, len(unique_labels))]
    # for label, color in zip(unique_labels, colors):
    #     if label == -1:
    #         continue
    #     cluster_point = sampled_points_xy[labels == label]
    #     # vote class
    #     cluster_classes = sampled_points_label[labels == label]
    #     unique, counts = np.unique(cluster_classes, return_counts=True)
    #     cluster_class = unique[np.argmax(counts)]
    #     if cluster_class == 0:
    #         continue
    #     draw.ax3.scatter(cluster_point[:, 0], cluster_point[:, 1], color = color_polate_4cls[cluster_class])
    #     ############
    #     # poly fit #
    #     ############
    #     order = 2
    #     # print(cluster_point[:, 0])
    #     poly_parameter = np.polyfit(cluster_point[:, 0], cluster_point[:, 1], order)
    #     poly = np.poly1d(poly_parameter)
    #     x_points = np.linspace(0, pred_seg.shape[2]-1, 100)
    #     draw.ax4.plot(cluster_point[:, 0], poly(cluster_point[:, 0]), color = color_polate_4cls[cluster_class], linewidth = 2)

    ###########################
    # get local max & cluster #
    ###########################
    tic = time.time()
    y_value = pred_seg.shape[1]-1
    # sample point contaion : (x, y, vector, label)
    sampled_points_info = []
    while(y_value > 0):
        # window = signal.general_gaussian(51, p=0.5, sig=20)
        # filtered = signal.fftconvolve(window, prob_map[y_value])
        # # draw.ax3.plot(filtered)
        # print(filtered)
        # print(filtered.shape)
        # filtered = (np.average(prob_map[y_value]) / np.average(filtered)) * filtered
        # prob_map[y_value] = np.roll(filtered, -25)
        # # print("==========================================")
        # print("number of cluster: {}".format(len(sampled_points_info)))
        indexes = signal.argrelextrema(prob_map[y_value], comparator=np.greater, order=2)
        if len(indexes[0]) == 0:
            y_value -= pick_interval
            continue
        if len(sampled_points_info) == 0:
            # for the first point, the angle can only be none
            sampled_points_info = [[[point_x, y_value, None, pred_seg_max[y_value, point_x]]] for point_x in indexes[0]]
            y_value -= pick_interval
            continue
        for point_x in indexes[0]:
            # print("-------------------------")
            # store [distance, direction]
            new_info = [[point_distance((cluster[-1][0], cluster[-1][1]), (point_x, y_value)), \
                 point_degree(cluster[-1][2], (point_x - cluster[-1][0], y_value - cluster[-1][1]))] for cluster in sampled_points_info]
            # print("new_info", new_info)
            new_info = np.array(new_info)
            min_distance_index = np.argmin(new_info[:, 0])
            min_distance = np.min(new_info[:, 0])
            angle = new_info[min_distance_index, 1]
            # print("min_distance", min_distance)
            # print("min_distance_index", min_distance_index)
            # print("angle", angle)
            # if min_distance < 50 and angle < 120:
            if min_distance < min_distance_threshold:
                new_vector = (point_x - sampled_points_info[min_distance_index][-1][0], y_value - sampled_points_info[min_distance_index][-1][1])
                sampled_points_info[min_distance_index].append([point_x, y_value, new_vector, pred_seg_max[y_value, point_x]])
            else:
                # new cluster
                # print("new cluster********")
                sampled_points_info.append([[point_x, y_value, None, pred_seg_max[y_value, point_x]]])
            # print("sampled_points_info", sampled_points_info)

        if len(sampled_points_info) != 0:
            for cluster in sampled_points_info:
                draw_info = np.array(cluster)
                draw.ax2.scatter(draw_info[:,0], draw_info[:, 1], color = "r")
        y_value -= pick_interval
        # plt.draw()
        # plt.waitforbuttonpress()

        # plt.draw()
        # plt.waitforbuttonpress()
    # print("Number of cluster: {}".format(len(sampled_points_info)))
    for cluster in sampled_points_info:
        cluster = np.array(cluster)
        if cluster.shape[0] < 5:
            continue
        # vote using class label
        unique, count = np.unique(cluster[:, -1], return_counts = True)
        cluster_class = unique[np.argmax(count)]
        if cluster_class == 0:
            continue
        # print("cluster", cluster.shape)
        cluster = cluster[cluster[:,-1] == cluster_class]    
        # print(cluster.shape)
        draw.ax3.scatter(cluster[:, 0], cluster[:, 1], color = color_polate_4cls[cluster_class])
        ############
        # poly fit #
        ############
        order = 2
        cluster_xy = cluster[:,:2] 
        cluster_xy = cluster_xy.astype(np.int32)
        poly_parameter = np.polyfit(cluster_xy[:,0], cluster_xy[:, 1], order)
        poly = np.poly1d(poly_parameter)
        # x_points = np.linspace(0, pred_seg.shape[2]-1, 100)
        draw.ax4.plot(cluster_xy[:, 0], poly(cluster_xy[:, 0]), color = color_polate_4cls[cluster_class], linewidth = 2)
        y_value -= pick_interval
    toc = time.time()

    toc =time.time()
    print("DBSCAN time: {}".format(toc - tic))
    tic =time.time()
    # set limits
    height, width = prob_map.shape[0], prob_map.shape[1]
    draw.ax.set_xlim(0.0, args.input_size[1])
    draw.ax2.set_xlim(0.0, width)
    draw.ax3.set_xlim(0.0, width)
    draw.ax4.set_xlim(0.0, width)

    draw.ax.set_ylim(args.input_size[0], 0.0)
    draw.ax2.set_ylim(height, 0.0)
    draw.ax3.set_ylim(height, 0.0)
    draw.ax4.set_ylim(height, 0.0)
    toc =time.time()
    # set titles
    draw.ax.set_title("Segmentation Result")
    draw.ax2.set_title("Local Maximum")
    draw.ax3.set_title("Clustering")
    draw.ax4.set_title("Polyfit")
    print("Set axes limit time: {}".format(toc - tic))

    plt.draw()
    plt.waitforbuttonpress()
    if writer is not None:
        writer.grab_frame()

def lane_post_process_find_mid(pred_seg, prob_map, pred_seg_max, img_draw, draw, writer):
    """Self define cluster method, try to find the mid points of each row"""
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
        print("=========================")
        #####################
        # get local maximum #
        #####################
        ticc = time.time()
        difference = np.diff(prob_map[y_value])
        # print(np.unique(difference), difference.shape)
        difference[difference < 0] = -difference[difference < 0]
        difference[difference < confidence_threshold-0.1] = 0
        where = np.where(difference)
        indexes = where[0][1::2] + where[0][::2]
        indexes = indexes//2
        """ plot """
        # draw.ax3.plot(prob_map[y_value], color = "b")
        # if where[0].shape != 0:
        #     for xc in where[0]:
        #         draw.ax3.axvline(x=xc, color = "r")
        # plt.draw()
        # plt.waitforbuttonpress()
        # draw.ax3.clear()

        """ plot """
        # if indexes.shape != 0:
        #     for xc in indexes:
        #         draw.ax3.axvline(x=xc, color = "r")
        # plt.draw()
        # plt.waitforbuttonpress()
        # draw.ax3.clear()
        # print(where)
        # print(indexes)


        # indexes = signal.argrelextrema(prob_map[y_value], comparator=np.greater, order=2)
        # indexes = signal.argrelextrema(new_data, comparator=np.greater, order=2)
        tocc = time.time()
        # print("***find_mid: {}".format(tocc-ticc))

        if indexes.shape == 0:
            y_value -= pick_interval
            continue
        if len(sampled_points_info) == 0:
            # for the first point, the angle can only be none
            # point: [x, y, angle, label]
            sampled_points_info = [[[point_x, y_value, None, pred_seg_max[y_value, point_x]]] for point_x in indexes]
            y_value -= pick_interval
            continue
        for point_x in indexes:
            print("-------------------------------------")
            # [distance, direction]
            new_info = [[point_distance((cluster[-1][0], cluster[-1][1]), (point_x, y_value)), \
                 point_degree(cluster[-1][2], (point_x - cluster[-1][0], y_value - cluster[-1][1]))] for cluster in sampled_points_info]
            new_info = np.array(new_info)
            # print(new_info)
            # """ min distance """
            min_distance_index = np.argmin(new_info[:, 0])
            min_distance = np.min(new_info[:, 0])
            angle = new_info[min_distance_index, 1]
            print("angle min_distance", angle, min_distance)

            # if (min_distance < min_distance_threshold):
            if (min_distance < min_distance_threshold and angle is None) or\
               (min_distance < min_distance_threshold and angle < 30 and angle is not None):
                print("add to cluster")
                new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_distance_index][-1][0], sampled_points_info[min_distance_index][-1][1]))
                if sampled_points_info[min_distance_index][-1][2] is not None:
                  average_vector = ((new_unit_vector[0]+sampled_points_info[min_distance_index][-1][2][0])/2,
                                    (new_unit_vector[1]+sampled_points_info[min_distance_index][-1][2][1])/2 )
                  sampled_points_info[min_distance_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
                else:
                    sampled_points_info[min_distance_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            else:
                # new cluster
                # print("new cluster********")
                sampled_points_info.append([[point_x, y_value, None, pred_seg_max[y_value, point_x]]])

            # """ min angle """
            # min_distance_index = np.argmin(new_info[:, 0])
            # min_distance = np.min(new_info[:, 0])
            # angle = new_info[min_distance_index, 1]

            # min_angle_index = np.argmin(new_info[:, 1])
            # min_angle = np.min(new_info[:, 1])
            # distance = new_info[min_angle_index, 0]
            # print("min_angle distance", min_angle, distance)
            # print("min_distance angle", min_distance, angle)

            # if(min_angle is not None and min_angle < 15 and distance < min_distance_threshold):
            #     print("***** add to cluster due to angle")
            #     new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_angle_index][-1][0], sampled_points_info[min_angle_index][-1][1]))
            #     if sampled_points_info[min_angle_index][-1][2] is not None:
            #       average_vector = ((new_unit_vector[0]+sampled_points_info[min_angle_index][-1][2][0])/2,
            #                         (new_unit_vector[1]+sampled_points_info[min_angle_index][-1][2][1])/2 )
            #       sampled_points_info[min_angle_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
            #     else:
            #         sampled_points_info[min_angle_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            # # elif (min_angle is not None and min_distance < min_distance_threshold and angle < angle_at_least):
            # #     print(" ***** add to cluster due to distance")
            # #     new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_distance_index][-1][0], sampled_points_info[min_distance_index][-1][1]))
            # #     if sampled_points_info[min_distance_index][-1][2] is not None:
            # #       average_vector = ((new_unit_vector[0]+sampled_points_info[min_distance_index][-1][2][0])/2,
            # #                         (new_unit_vector[1]+sampled_points_info[min_distance_index][-1][2][1])/2 )
            # #       sampled_points_info[min_distance_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
            # #     else:
            # #         sampled_points_info[min_distance_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            # elif (min_distance < min_distance_threshold and min_angle is None):
            #     print(" ***** add to cluster due to distance")
            #     new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_distance_index][-1][0], sampled_points_info[min_distance_index][-1][1]))
            #     if sampled_points_info[min_distance_index][-1][2] is not None:
            #       average_vector = ((new_unit_vector[0]+sampled_points_info[min_distance_index][-1][2][0])/2,
            #                         (new_unit_vector[1]+sampled_points_info[min_distance_index][-1][2][1])/2 )
            #       sampled_points_info[min_distance_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
            #     else:
            #         sampled_points_info[min_distance_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            # else:
            #     # new cluster
            #     print("****** new cluster")
            #     sampled_points_info.append([[point_x, y_value, None, pred_seg_max[y_value, point_x]]])        


            # new_info = [[point_distance((cluster[-1][0], cluster[-1][1]), (point_x, y_value)), \
            #      point_degree(cluster[-1][2], (point_x - cluster[-1][0], y_value - cluster[-1][1])), \
            #      point_info(point_distance((cluster[-1][0], cluster[-1][1]), (point_x, y_value)),\
            #                 point_degree(cluster[-1][2], (point_x - cluster[-1][0], y_value - cluster[-1][1])))] for cluster in sampled_points_info]
            # # print("new_info", new_info)
            # new_info = np.array(new_info)
            # """ min info """
            # min_distance_index = np.argmin(new_info[:, 0])
            # min_distance = np.min(new_info[:, 0])
            # angle = new_info[min_distance_index, 1]

            # min_angle_index = np.argmin(new_info[:, 1])
            # min_angle = np.min(new_info[:, 1])
            # distance = new_info[min_angle_index, 0]
            # print("min_angle distance", min_angle, distance)
            # print("min_distance angle", min_distance, angle)

            # if(min_angle is not None and min_angle < 15 and distance < min_distance_threshold):
            #     print("***** add to cluster due to angle")
            #     new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_angle_index][-1][0], sampled_points_info[min_angle_index][-1][1]))
            #     if sampled_points_info[min_angle_index][-1][2] is not None:
            #       average_vector = ((new_unit_vector[0]+sampled_points_info[min_angle_index][-1][2][0])/2,
            #                         (new_unit_vector[1]+sampled_points_info[min_angle_index][-1][2][1])/2 )
            #       sampled_points_info[min_angle_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
            #     else:
            #         sampled_points_info[min_angle_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            # # elif (min_angle is not None and min_distance < min_distance_threshold and angle < angle_at_least):
            # #     print(" ***** add to cluster due to distance")
            # #     new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_distance_index][-1][0], sampled_points_info[min_distance_index][-1][1]))
            # #     if sampled_points_info[min_distance_index][-1][2] is not None:
            # #       average_vector = ((new_unit_vector[0]+sampled_points_info[min_distance_index][-1][2][0])/2,
            # #                         (new_unit_vector[1]+sampled_points_info[min_distance_index][-1][2][1])/2 )
            # #       sampled_points_info[min_distance_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
            # #     else:
            # #         sampled_points_info[min_distance_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            # elif (min_distance < min_distance_threshold and min_angle is None):
            #     print(" ***** add to cluster due to distance")
            #     new_unit_vector = get_unit_vector((point_x, y_value),(sampled_points_info[min_distance_index][-1][0], sampled_points_info[min_distance_index][-1][1]))
            #     if sampled_points_info[min_distance_index][-1][2] is not None:
            #       average_vector = ((new_unit_vector[0]+sampled_points_info[min_distance_index][-1][2][0])/2,
            #                         (new_unit_vector[1]+sampled_points_info[min_distance_index][-1][2][1])/2 )
            #       sampled_points_info[min_distance_index].append([point_x, y_value, average_vector, pred_seg_max[y_value, point_x]])
            #     else:
            #         sampled_points_info[min_distance_index].append([point_x, y_value, new_unit_vector, pred_seg_max[y_value, point_x]])
            # else:
            #     # new cluster
            #     print("****** new cluster")
            #     sampled_points_info.append([[point_x, y_value, None, pred_seg_max[y_value, point_x]]])   

        print(len(sampled_points_info))
        # print("sampled_points_info", sampled_points_info)
        print("")
        # plt.draw()
        # plt.waitforbuttonpress()
        if len(sampled_points_info) != 0:
            for cluster in sampled_points_info:
                draw_info = np.array(cluster)
                draw.ax2.scatter(draw_info[:,0], draw_info[:, 1], color = "r")
        y_value -= pick_interval
        # plt.draw()
        # plt.waitforbuttonpress()
    # print("Number of cluster: {}".format(len(sampled_points_info)))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1.0, len(sampled_points_info))]
    for i, cluster in enumerate(sampled_points_info):
        cluster = np.array(cluster)
        if cluster.shape[0] < 5:
            continue
        # vote using class label

        unique, count = np.unique(cluster[:, -1], return_counts = True)
        cluster_class = unique[np.argmax(count)]
        if cluster_class == 0:
            continue
        # print("cluster", cluster.shape)
        cluster = cluster[cluster[:,-1] == cluster_class]    
        # print(cluster.shape)
        # draw.ax3.scatter(cluster[:, 0], cluster[:, 1], color = color_polate_4cls[cluster_class])
        draw.ax3.scatter(cluster[:, 0], cluster[:, 1], color = colors[i])
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
        if np.where(x_cor_diff < 0)[0].shape[0] == 0:
        # if True:
            """case to use poly fit"""
            poly_parameter = np.polyfit(cluster_xy[:,0], cluster_xy[:, 1], order)
            poly = np.poly1d(poly_parameter)
            # x_points = np.linspace(0, pred_seg.shape[2]-1, 100)
            draw.ax4.plot(cluster_xy[:, 0], poly(cluster_xy[:, 0]), color = color_polate_4cls[cluster_class], linewidth = 2)
        else:
            draw.ax4.plot(cluster_xy[:, 0], cluster_xy[:, 1], color = color_polate_4cls[cluster_class], linewidth = 2)
        y_value -= pick_interval
    toc = time.time()
    print("DBSCAN time: {}".format(toc - tic))
    tic =time.time()
    # set limits
    height, width = prob_map.shape[0], prob_map.shape[1]
    draw.ax.set_xlim(0.0, args.input_size[1])
    draw.ax2.set_xlim(0.0, width)
    draw.ax3.set_xlim(0.0, width)
    draw.ax4.set_xlim(0.0, width)

    draw.ax.set_ylim(args.input_size[0], 0.0)
    draw.ax2.set_ylim(height, 0.0)
    draw.ax3.set_ylim(height, 0.0)
    draw.ax4.set_ylim(height, 0.0)
    toc =time.time()
    # set titles
    draw.ax.set_title("Segmentation Result")
    draw.ax2.set_title("Local Maximum")
    draw.ax3.set_title("Clustering")
    draw.ax4.set_title("Polyfit")
    print("Set axes limit time: {}".format(toc - tic))

    plt.draw()
    plt.waitforbuttonpress()
    if writer is not None:
        writer.grab_frame()

def demo_lane(net, dataset, logger, draw, writer, engine):
    image_list = dataset.data_list
    net.target_available = False
    skip_frame = 100
    count = 0
    net.eval()
    dataset_path = os.path.join("..", "datasets", "bdd100k", "bdd100k")
    # define drawer
    with torch.no_grad():
        for image in image_list:
                # if count < 2500 or count > 7000:
            #     continue
            image_path = os.path.join(dataset_path, "images", "100k", "val", image+".jpg")
            print(image_path)
            frame = cv2.imread(image_path)
            start = time.time()
            tic =time.time()
            # plt.cla()
            draw.ax.clear()
            draw.ax2.clear()
            draw.ax3.clear()
            draw.ax4.clear()
            toc = time.time()
            print("Clear axes time: {:.3f}".format(toc - tic))
            # draw2.ax.clear()
            # draw2.ax2.clear()
            tic = time.time()
            # resize image
            img = cv2.resize(frame, (dataset.input_size[1], dataset.input_size[0]), 0, 0, interpolation = cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            # BGR to RGB
            img = img[:,:,[ 2, 1, 0]]
            # transfrom and expand dim
            input_ = img.transpose((2, 0, 1))
            input_ = np.expand_dims(input_, axis = 0)
            if DEFINE_ONNX:
                print("ONNX")
                toc = time.time()
                print("Preprocess time: {:.3f}".format(toc - tic))

                tic = time.time()
                pred_seg = engine.run(input_)[0]
                print("Net forward time: {}".format(toc - tic))
                toc =time.time()
            else:
                print("Not ONNX")
                # push to tensor
                input_ = torch.from_numpy(input_).float()
                input_ = input_.to(device)
                toc = time.time()
                print("Preprocess time: {:.3f}".format(toc - tic))
                # net forward
                tic =time.time()
                pred_seg = net(input_)
                pred_seg = softmax(pred_seg)
                toc =time.time()
                print("Net forward time: {}".format(toc - tic))
                tic =time.time()
                # threhold
                pred_seg[pred_seg < confidence_threshold] = 0
                # back to cpu
                pred_seg = pred_seg.data.cpu().numpy()
            
            # squeeze
            pred_seg = np.squeeze(pred_seg, axis = 0)
            # image copy for drawing
            # img_seg = img.copy()
            # img_seg2 = img.copy()
            # img_seg = img_seg.astype(np.uint8)
            # img_seg2 = img_seg2.astype(np.uint8)
            # img_draw = cv2.resize(img, (pred_seg.shape[2], pred_seg.shape[1]), 0, 0, interpolation = cv2.INTER_LINEAR)
            # img_draw2 = cv2.resize(img, (pred_seg.shape[2], pred_seg.shape[1]), 0, 0, interpolation = cv2.INTER_LINEAR)
            img_draw = img.copy()
            img_draw2 = img.copy()
            img_draw = img_draw.astype(np.uint8)
            img_draw2 = img_draw2.astype(np.uint8)
            # binary map 
            # binary_map = pred_seg.copy()
            # binary_map = binary_map.astype(np.bool)
            # prob map
            prob_map = np.max(pred_seg[3:, :, :], axis = 0)
            toc =time.time()
            print("Pre-Post-process time: {}".format(toc - tic))
            # segmentation prediction
            pred_seg_max = np.argmax(pred_seg, axis = 0)
            # squeeze dim
            pred_seg_max = np.squeeze(pred_seg_max)
            # draw segmentation result
            if SHOW_SEG_RESULT:
                tic = time.time()
                # upsample back if prediction size not equal to input_size
                pred_seg_max_up = upsample_back(pred_seg_max, dataset.input_size)
                # RBG for color 
                # print(pred_seg_max.shape)
                RGB = np.zeros((dataset.input_size[0], dataset.input_size[1], 3))
                RGB2 = np.zeros((dataset.input_size[0], dataset.input_size[1], 3))
                # flag for overlay area
                overlay_flag = np.zeros((dataset.input_size[0], dataset.input_size[1]))
                overlay_flag2 = np.zeros((dataset.input_size[0], dataset.input_size[1]))
                # draw
                for key, color in cmap_4cls.items():
                # for key, color in dataset.color_map.items():
                    if key == 0:
                        continue
                    if key == 0 or key >= 3:
                        RGB2[pred_seg_max_up == key] = np.array(color[::-1])
                        overlay_flag2[pred_seg_max_up == key] = 1
                        continue
                    else:
                        RGB2[pred_seg_max_up == key] = np.array(color[::-1])
                        overlay_flag2[pred_seg_max_up == key] = 1
                        RGB[pred_seg_max_up == key] = np.array(color[::-1])
                        overlay_flag[pred_seg_max_up == key] = 1
                # overlay
                RGB = RGB.astype(np.uint8)
                RGB2 = RGB2.astype(np.uint8)
                alpha = 0.5
                overlay = cv2.addWeighted(img_draw, alpha, RGB, (1-alpha), 0)
                overlay2 = cv2.addWeighted(img_draw2, alpha, RGB2, (1-alpha), 0)
                img_draw[overlay_flag == 1] = overlay[overlay_flag == 1]
                img_draw2[overlay_flag2 == 1] = overlay2[overlay_flag2 == 1]
                # plot it on axes
                draw.ax.imshow(img_draw2)

                # draw it on screen
                # plt.draw()
                # plt.waitforbuttonpress()
                # plt.show
                # plt.pause(1)
                # cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
                # cv2.imshow("overlay", img_draw)
                # cv2.waitKey(0)
                img_draw_down = cv2.resize(img_draw, (pred_seg.shape[2], pred_seg.shape[1]), 0, 0, interpolation = cv2.INTER_LINEAR)
                draw.ax2.imshow(img_draw_down)
                draw.ax3.imshow(img_draw_down)
                draw.ax4.imshow(img_draw_down)
                toc =time.time()
                print("Segmentation draw time: {}".format(toc - tic))
            # lane_post_process(pred_seg, None, prob_map, pred_seg_max, img_draw_down, draw, writer)
            # lane_post_process_find_mid(pred_seg, prob_map, pred_seg_max, img_draw, draw, writer)
            end = time.time()
            plt.draw()
            plt.waitforbuttonpress()
            # if writer is not None:
            #     writer.grab_frame()
            print("Total time: {}".format(end - start))
        

if __name__ == "__main__":
    args = get_arguments()
    testing_log = "{}_{}x{}_demo_lane_line".format(args.model, args.input_size[0], args.input_size[1])
    logger = MyLog(testing_log)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logger.info(testing_log)
    logger.info(args)

    # # input video capture
    # assert os.path.exists(args.input_video), "Input video {} does not exist.".format(args.input_video)
    # cap = cv2.VideoCapture(args.input_video)
    # output video
    writer = None
    output_path = None
    if args.output_video:
        output_dir = os.path.join("output_video")
        if not os.path.exists(output_dir):
            os.makedirs("output_video")
        metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
        writer = FFMpegWriter(fps=25, metadata=metadata, bitrate = -1, codec="libx264")
        output_path = os.path.join(output_dir, testing_log + "_" + args.input_video.split("/")[-1][:-4]+"_lane_line_seg.mp4")

    # dataset
    split = "val"
    lib = importlib.import_module("dataset.bdd100kLaneandLineSeg")
    test_dataset = lib.bdd100kLandandLineDataset(input_size = args.input_size, split = split)

    # load onnx model
    engine = None
    if DEFINE_ONNX:
        model = onnx.load("test.onnx")
        engine = backend.prepare(model, device='CUDA:0')

    # network
    net = net_option(name = args.model, mode = "segmentation")
    net = net.to(device)
    num_parameters = count_parameters(net)
    logger.info("Number of network parameters: {}".format(num_parameters))

    # resume from checkpoint

    assert os.path.exists(args.checkpoint), "Checkpoint {} does not exist.".format(args.checkpoint)
    state = torch.load(args.checkpoint)
    net.load_state_dict(state["model_state"], strict = True)
    logger.info("Resume from previous model {}".format(args.checkpoint))
    # demo
    drawer = drawer()
    if writer is not None:
        print(output_path)
        with writer.saving(draw.fig, output_path, 100):
            demo_lane(net, test_dataset, logger, draw, writer, engine)
    else:
        demo_lane(net, test_dataset, logger, drawer, writer, engine)
    print(output_path)







