#############################################################
# Script for lane and line segmentation and object detection demo.
# Model was trained on BDD100K and use for video demo.
# Show the lane segmentation result and post-process results
# Remove all redudant
# Date: 20190412
#############################################################
from __future__ import print_function, division
import os
import sys
ROOT_DIR = os.path.realpath(__file__).split("demo/demo_det_seg_lane_line_bdd_video_test_speed_multi.py")[0]
sys.path.append(ROOT_DIR)
import numpy as np
import cv2
# from scipy.interpolate import interp1d
# import scipy.signal as signal
# from sklearn.cluster import DBSCAN
import math
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

import time
import importlib
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network.net import net_option
from utils.logger import MyLog
from utils.core_utils import count_parameters
# from dataset.augmentation_seg import bdd100kLandSegAug
import threading
import multiprocessing

softmax = torch.nn.Softmax(dim = 1)
SHOW_SEG_RESULT = True
confidence_threshold = 0.7
DEBUG_MIN_OBJ_WIDTH = True
wmin = 100
pg.setConfigOptions(imageAxisOrder = "row-major")
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

color_polate_4cls_QT = {3: "FF0",
                        4: "0FF",
                        5: "F00"}


cmap_4cls = {1: (  0,255,  0),
             2: (  0,  0,255),
             3: (255,255,  0),
             4: (  0,255,255),
             5: (255,  0,  0)}
alpha = 0.5
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest = "model", type = str, help = "Network archtecture.")
    parser.add_argument("-i", dest = "input_size", type = int, nargs = "+", default = [256,512], help = "Network input size.")
    parser.add_argument("-c", dest = "checkpoint", type = str, help = "Checkpoint file.")
    parser.add_argument("-iv", dest = "input_video", type = str, required = True, help = "Input video for demo.")
    parser.add_argument("-ot", dest = "od_threshold",  type = float, default = 0.3, help = "Detection Confidence threshold.")
    parser.add_argument("-ov", dest = "output_video", action = "store_true", help = "Input video for demo.")
    return parser.parse_args()


class App(QtGui.QMainWindow):
    # def __init__(self, input_video, net, test_dataset, logger, parent=None):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        # #### input args #########
        # self._input_video = input_video
        # self._net = net
        # self._dataset = test_dataset
        # self._logger = logger

        # ### net setting ###
        # self._net.target_available = False
        # self._net.eval()
        # self._softmax = torch.nn.Softmax(dim = 1)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.resize(100, 100)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox(invertY = True)
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 1024, 512))

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        #  curve on image
        self.curve1 = pg.PlotCurveItem()
        self.view.addItem(self.curve1)
        self.curve2 = pg.PlotCurveItem()
        self.view.addItem(self.curve2)
        self.curve3 = pg.PlotCurveItem()
        self.view.addItem(self.curve3)
        self.curve4 = pg.PlotCurveItem()
        self.view.addItem(self.curve4)
        self.curve5 = pg.PlotCurveItem()
        self.view.addItem(self.curve5)
        self.curve6 = pg.PlotCurveItem()
        self.view.addItem(self.curve6)


        # self.canvas.nextRow()
        #  line plot
        # self.otherplot = self.canvas.addPlot()
        # self.h2 = self.otherplot.plot(pen='y')


        #### opencv capture ######
        # input video capture
        # assert os.path.exists(self._input_video), "Input video {} does not exist.".format(self._input_video)
        # self._cap = cv2.VideoCapture(self._input_video)
        # self._skip_frame = 100
        # self._video_count = 0

        #### Set Data  #####################
        # self.x = np.linspace(0,50., num=100)
        # self.X,self.Y = np.meshgrid(self.x,self.x)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()


        #### Start  #####################
        # self.main()
        # self._update()

    def _update(self):
        global q
        # ret, frame = self._cap.read()
        item = q.get()
        self.img.setImage(item[0], autoDownsample = False)
        # for i, content in enumerate(item):
        for i in range(7):
            draw_flag = False
            if i == 0:
                continue
            if len(item) > i:
                draw_flag = True
            if i == 1 and draw_flag:
                ppen = pg.mkPen(color= color_polate_4cls_QT[item[i][2]], width= 2)
                self.curve1.setData(item[i][0], item[i][1], pen=ppen)
            elif i == 1 and not draw_flag:
                self.curve1.setData(np.array([np.nan]), np.array([np.nan]), pen=1)
            elif i == 2 and draw_flag:
                ppen = pg.mkPen(color= color_polate_4cls_QT[item[i][2]], width= 2)
                self.curve2.setData(item[i][0], item[i][1], pen=ppen)
            elif i == 2 and not draw_flag:
                self.curve2.setData(np.array([np.nan]), np.array([np.nan]), pen=1)
            elif i == 3 and draw_flag:
                ppen = pg.mkPen(color= color_polate_4cls_QT[item[i][2]], width= 2)
                self.curve3.setData(item[i][0], item[i][1], pen=ppen)
            elif i == 3 and not draw_flag:
                self.curve3.setData(np.array([np.nan]), np.array([np.nan]), pen=1)
            elif i == 4 and draw_flag:
                ppen = pg.mkPen(color= color_polate_4cls_QT[item[i][2]], width= 2)
                self.curve4.setData(item[i][0], item[i][1], pen=ppen)
            elif i == 4 and not draw_flag:
                self.curve4.setData(np.array([np.nan]), np.array([np.nan]), pen=1)
            elif i == 5 and draw_flag:
                ppen = pg.mkPen(color= color_polate_4cls_QT[item[i][2]], width= 2)
                self.curve5.setData(item[i][0], item[i][1], pen=ppen)
            elif i == 5 and not draw_flag:
                self.curve5.setData(np.array([np.nan]), np.array([np.nan]), pen=1)
            elif i == 6 and draw_flag:
                ppen = pg.mkPen(color= color_polate_4cls_QT[item[i][2]], width= 2)
                self.curve6.setData(item[i][0], item[i][1], pen=ppen)
            elif i == 6 and not draw_flag:
                self.curve6.setData(np.array([np.nan]), np.array([np.nan]), pen=1)

        
        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)
        # QtCore.QTimer.singleShot(1, self._update(queue))
        self.counter += 1

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

def lane_post_process_find_mid(pred_seg, prob_map, pred_seg_max, img_draw, q):
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
               (min_distance < min_distance_threshold and angle < 30 and angle is not None):
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
            # draw.ax4.plot(cluster_xy[:, 0], poly(cluster_xy[:, 0]), color = color_polate_4cls[cluster_class], linewidth = 2)
        else:
            x, y = cluster_xy[:, 0], cluster_xy[:, 1]
            # draw.ax4.plot(cluster_xy[:, 0], cluster_xy[:, 1], color = color_polate_4cls[cluster_class], linewidth = 2)
        # if i == 0:
        #     ppen = pg.mkPen(color= color_polate_4cls_QT[cluster_class], width= 2)
        #     self.curve1.setData(x,y,pen=ppen)
        # elif i == 1:
        #     ppen = pg.mkPen(color= color_polate_4cls_QT[cluster_class], width= 2)
        #     self.curve2.setData(x,y,pen=ppen)
        # elif i == 2:
        #     ppen = pg.mkPen(color= color_polate_4cls_QT[cluster_class], width= 2)
        #     self.curve3.setData(x,y,pen=ppen)
        # elif i == 3:
        #     ppen = pg.mkPen(color= color_polate_4cls_QT[cluster_class], width= 2)
        #     self.curve4.setData(x,y,pen=ppen)
        # elif i == 4:
        #     ppen = pg.mkPen(color= color_polate_4cls_QT[cluster_class], width= 2)
        #     self.curve5.setData(x,y,pen=ppen)
        # elif i == 5:
        #     ppen = pg.mkPen(color= color_polate_4cls_QT[cluster_class], width= 2)
        #     self.curve6.setData(x,y,pen=ppen)
        queue_info.append([x, y, cluster_class])
        y_value -= pick_interval
    toc = time.time()
    print("DBSCAN time: {}".format(toc - tic))
    queue_info = [img_draw] + queue_info
    q.put(queue_info)
    # self.img.setImage(img_draw, autoDownsample = False)

    
def main_process(cap, dataset, softmax, q):
    # RGB = np.zeros((dataset.input_size[0], dataset.input_size[1], 3), dtype = np.uint8)
    skip_frame = 1
    count = 0
    net.target_available = False
    net.eval()
    # print(q,cap,dataset)
    with torch.no_grad():
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                if count % skip_frame != 0:
                    count += 1
                    continue
                else:
                    count += 1
                RGB = np.zeros((dataset.input_size[0], dataset.input_size[1], 3), dtype = np.uint8)
                print("--------------------------")
                start = time.time()
                tic = time.time()
            
                # resize image
                img = cv2.resize(frame, (dataset.input_size[1], dataset.input_size[0]), 0, 0, interpolation = cv2.INTER_LINEAR)
                # img = img.astype(np.float32)
                # BGR to RGB
                img = img[:,:,[ 2, 1, 0]]
                # transfrom and expand dim
                input = img.transpose((2, 0, 1))
                input = np.expand_dims(input, axis = 0)
                # push to tensor
                input = torch.from_numpy(input).float()
                input = input.to(device)
                toc = time.time()
                print("Preprocess time: {:.3f}".format(toc - tic))
                # net forward
                tic =time.time()
                scores, classification, transformed_anchors, pred_seg_t = net(input)
                pred_seg = softmax(pred_seg_t)
                toc =time.time()
                print("Net forward time: {}".format(toc - tic))
                tic =time.time()
                # threhold
                # pred_seg[pred_seg < confidence_threshold] = 0
                # back to cpu
                pred_seg = pred_seg.squeeze(dim = 0)
                pred_seg = pred_seg.data.cpu().numpy()
                pred_seg[pred_seg < confidence_threshold] = 0
                # squeeze
                # pred_seg = np.squeeze(pred_seg, axis = 0)
                # image copy for drawing
                # img_seg = img.copy()
                # img_seg2 = img.copy()
                # img_seg = img_seg.astype(np.uint8)
                # img_seg2 = img_seg2.astype(np.uint8)
                # img_draw = cv2.resize(img, (pred_seg.shape[2], pred_seg.shape[1]), 0, 0, interpolation = cv2.INTER_LINEAR)
                # img_draw2 = cv2.resize(img, (pred_seg.shape[2], pred_seg.shape[1]), 0, 0, interpolation = cv2.INTER_LINEAR)
                # img_draw = img.copy()
                # img_draw = img
                img_draw = img.astype(np.uint8)
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
                # pred_seg_max = np.squeeze(pred_seg_max)
                # draw segmentation result
                if True:
                    tic = time.time()
                    # upsample back if prediction size not equal to input_size
                    pred_seg_max_up = upsample_back(pred_seg_max, dataset.input_size)
                    # RBG for color 
                    # print(pred_seg_max.shape)
                    # RGB2 = np.zeros((dataset.input_size[0], dataset.input_size[1], 3))
                    # flag for overlay area
                    # overlay_flag2 = np.zeros((dataset.input_size[0], dataset.input_size[1]))
                    overlay_flag = np.zeros((dataset.input_size[0], dataset.input_size[1]))
                    # draw
                    for key, color in cmap_4cls.items():
                    # for key, color in dataset.color_map.items():
                        if key == 0:
                            continue
                        if key >= 3:
                            # RGB2[pred_seg_max_up == key] = np.array(color[::-1])
                            # overlay_flag2[pred_seg_max_up == key] = 1
                            continue
                        else:
                            # RGB2[pred_seg_max_up == key] = np.array(color[::-1])
                            # overlay_flag2[pred_seg_max_up == key] = 1
                            RGB[pred_seg_max_up == key] = np.array(color)
                            overlay_flag[pred_seg_max_up == key] = 1
                    # overlay
                    # RGB = RGB.astype(np.uint8)
                    # RGB2 = RGB2.astype(np.uint8)
                    overlay = cv2.addWeighted(img_draw, alpha, RGB, (1-alpha), 0)
                    # overlay2 = cv2.addWeighted(img_draw2, alpha, RGB2, (1-alpha), 0)
                    img_draw[overlay_flag == 1] = overlay[overlay_flag == 1]
                    # img_draw2[overlay_flag2 == 1] = overlay2[overlay_flag2 == 1]
                    # plot it on axes
                    #### draw.ax.imshow(img_draw)
                    # draw it on screen
                    # plt.draw()
                    # plt.waitforbuttonpress()
                    # plt.show
                    # plt.pause(1)
                    # cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
                    # cv2.imshow("overlay", img_draw)
                    # cv2.waitKey(0)
                    # img_draw_down = cv2.resize(img_draw, (pred_seg.shape[2], pred_seg.shape[1]), 0, 0, interpolation = cv2.INTER_LINEAR)
                    # img_draw_down = overlay
                    img_draw_down = img_draw.copy()
                    
                    toc =time.time()
                    print("Segmentation draw time: {}".format(toc - tic))
                ##################
                # draw detection #
                ##################
                tic = time.time()
                scores = scores.data.cpu().numpy()
                idxs = np.where(scores > args.od_threshold) 
                for j in range(idxs[0].shape[0]):
                    class_ =int(classification[idxs[0][j]])
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(round(bbox[0]))
                    y1 = int(round(bbox[1]))
                    x2 = int(round(bbox[2]))
                    y2 = int(round(bbox[3]))
                    cv2.rectangle(img_draw_down, (x1,y1), (x2, y2),color = (0,0,0)) 
                toc = time.time()
                print("Detection draw took: {}".format(toc - tic))
                # self.lane_post_process(pred_seg, prob_map, pred_seg_max, img_draw_down)
                # img_draw_down = img_draw_down.T
        
                # self.img.setImage(img_draw_down, autoDownsample = False)
                lane_post_process_find_mid(pred_seg, prob_map, pred_seg_max, img_draw_down, q)
                end = time.time()
                print("Total time: {}".format(end - start))

def read_frame(cap, q):
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            print("read frameing")
            q.put([frame])
        # return frame

if __name__ == "__main__":
    args = get_arguments()
    testing_log = "{}_demo_lane_line".format(args.model)
    logger = MyLog(testing_log)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logger.info(testing_log)
    logger.info(args)

    # input video capture
    assert os.path.exists(args.input_video), "Input video {} does not exist.".format(args.input_video)
    cap = cv2.VideoCapture(args.input_video)

    # dataset
    split = "val"
    lib = importlib.import_module("dataset.bdd100kLaneandLineSeg")
    test_dataset = lib.bdd100kLandandLineDataset(input_size = args.input_size, split = split)

    # network
    net = net_option(name = args.model, mode = "end2end")
    net = net.to(device)
    num_parameters = count_parameters(net)
    logger.info("Number of network parameters: {}".format(num_parameters))

    # resume from checkpoint
    assert os.path.exists(args.checkpoint), "Checkpoint {} does not exist.".format(args.checkpoint)
    state = torch.load(args.checkpoint)
    net.load_state_dict(state["model_state"])
    logger.info("Resume from previous model {}".format(args.checkpoint))
    # demo

    # demo_lane(net, test_dataset, logger)
    app = QtGui.QApplication(sys.argv)
    # thisapp = App(args.input_video, net, test_dataset, logger)
    thisapp = App()
    thisapp.show()

    assert os.path.exists(args.input_video), "Input video {} does not exist.".format(args.input_video)
    _cap = cv2.VideoCapture(args.input_video)

    # v1
    q = multiprocessing.Queue()
    assert os.path.exists(args.input_video), "Input video {} does not exist.".format(args.input_video)
    cap = cv2.VideoCapture(args.input_video)
    # cap = cv2.VideoCapture(0)
    softmax = torch.nn.Softmax(dim = 1)
    t = threading.Thread(target=main_process, args=(cap, test_dataset,softmax, q))
    t.start()
    thisapp._update()

    # p = multiprocessing.Process(target=thisapp._update, args=(q, ))
    # p.start()
    # thisapp._update()
    print("Hello")
    sys.exit(app.exec_())
    t.join()







