#############################################################
# This script contain code to visualize detection result.
# Date: 20190605
#############################################################
from __future__ import print_function, division
import os
import sys
ROOT_DIR = os.path.realpath(__file__).split("demo/demo_det_video.py")[0]
sys.path.append(ROOT_DIR)
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as patches

import time
import importlib
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network.net import net_option
from utils.logger import MyLog
from utils.core_utils import count_parameters, upsample_back
from utils.draw_function import get_rect_patch
# from dataset.augmentation_seg import bdd100kLandSegAug
import lane_post_process.lane_line_postprocess as lane_line_postprocess

import lane_post_process.lane_postprocess as lane_postprocess
# import lane_post_process.lane_postprocess 

# nn function
# softmax = torch.nn.Softmax(dim = 1)
LINE_WIDTH = 3
dataset_type = "IVS3"

# show result
SHOW_BBOX_RESULT = True
SHOW_SEG_RESULT = True
SHOW_IPM_OBJECT = True
ipm_size = (1080, 800)
DEBUG_MIN_OBJ_WIDTH = True
wmin = 100
# threshold
confidence_threshold = 0.7

# pick interval: 512x1024:8 
y_offset = 8
min_distance_threshold = 100
min_angle_threshold = 60

cmap_4cls = {1: (  0,255,  0),
             2: (255,  0,  0),
             3: (  0,255,255),
             4: (255,255,  0),
             5: (  0,  0,255)}

class drawer(object):
    def __init__(self, mode = "normal"):
        self.mode = mode
        w, h = 16, 9
        self.fig = plt.figure(figsize = (w, h))
        self.ax  = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.ax5 = None
        self.draw_netout = None
        self.draw_local_maximum = None
        self.draw_clustering = None
        self.draw_poly_fit = None
        self.draw_IPM = None
        if self.mode == "normal":
            #left bottom width height
            self.ax  = self.fig.add_axes([0.05, 0.50, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
            self.ax2 = self.fig.add_axes([0.55, 0.50, 0.4, 0.4], frameon = True)  # If False, suppress drawing the figure frame.
            self.ax3 = self.fig.add_axes([0.05, 0.05, 0.4, 0.4], frameon = True)
            self.ax4 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4], frameon = True)
            self.custon_handle = custom_lines = [
                matplotlib.lines.Line2D([0], [0], color="#00FF00", lw=4),
                matplotlib.lines.Line2D([0], [0], color="#0000FF", lw=4),
                matplotlib.lines.Line2D([0], [0], color="#FF0000", lw=4),
                matplotlib.lines.Line2D([0], [0], color="#00FFFF", lw=4),
                matplotlib.lines.Line2D([0], [0], color="#FFFF00", lw=4)]
            self.fig.legend(self.custon_handle,
            labels=('Main Lane', 'Alter Lane', 'Single Line', 'Dashed Line', 'Double Line'),
            loc='upper right')
            self.draw_netout = self.ax
            self.draw_local_maximum = self.ax2
            self.draw_clustering = self.ax3
            self.draw_poly_fit = self.ax4
        elif self.mode == "IPM":
            self.ax  = self.fig.add_axes([0.05, 0.00, 0.4, 0.2], frameon = True)  # If False, suppress drawing the figure frame.
            self.ax2 = self.fig.add_axes([0.05, 0.25, 0.4, 0.2], frameon = True)  # If False, suppress drawing the figure frame.
            self.ax3 = self.fig.add_axes([0.05, 0.50, 0.4, 0.2], frameon = True)
            self.ax4 = self.fig.add_axes([0.05, 0.75, 0.4, 0.2], frameon = True)
            self.ax5 = self.fig.add_axes([0.55, 0.05, 0.4, 0.9], frameon = True)
            self.draw_netout = self.ax
            self.draw_local_maximum = self.ax2
            self.draw_clustering = self.ax3
            self.draw_poly_fit = self.ax4
            self.draw_IPM = self.ax5
        elif self.mode == "only_final":
            self.ax = self.fig.add_axes([0.00, 0.00, 1.0, 1.0], frameon = False)
            self.draw_poly_fit = self.ax

    def set_limit_titles(self, height, width):
        if self.mode == "normal":
            # set axes limits
            draw.ax.set_xlim(0.0, args.input_size[1])
            draw.ax2.set_xlim(0.0, width)
            draw.ax3.set_xlim(0.0, width)
            draw.ax4.set_xlim(0.0, width)
            draw.ax.set_ylim(args.input_size[0], 0.0)
            draw.ax2.set_ylim(height, 0.0)
            draw.ax3.set_ylim(height, 0.0)
            draw.ax4.set_ylim(height, 0.0)
            # set titles
            draw.ax.set_title("Segmentation Result")
            draw.ax2.set_title("Local Maximum")
            draw.ax3.set_title("Clustering")
            draw.ax4.set_title("Polyfit")
    
        elif self.mode == "IPM":
            draw.ax.set_xlim(0.0, args.input_size[1])
            draw.ax2.set_xlim(0.0, width)
            draw.ax3.set_xlim(0.0, width)
            draw.ax4.set_xlim(0.0, width)
            draw.ax5.set_xlim(0.0, ipm_size[1])
            draw.ax.set_ylim(args.input_size[0], 0.0)
            draw.ax2.set_ylim(height, 0.0)
            draw.ax3.set_ylim(height, 0.0)
            draw.ax4.set_ylim(height, 0.0)
            draw.ax5.set_ylim(ipm_size[0], 0.0)
            # set titles
            draw.ax.set_title("Segmentation Result")
            draw.ax2.set_title("Local Maximum")
            draw.ax3.set_title("Clustering")
            draw.ax4.set_title("Polyfit")
            draw.ax5.set_title("Bird View")
        elif self.mode == "only_final":
            # set axes limits
            draw.ax.set_xlim(0.0, width)
            draw.ax.set_ylim(height, 0.0)
            draw.ax.axis("off")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest = "model", type = str, help = "Network archtecture.")
    parser.add_argument("-i", dest = "input_size", type = int, nargs = "+", default = [512,1024], help = "Network input size.")
    parser.add_argument("-c", dest = "checkpoint", type = str, help = "Checkpoint file.")
    parser.add_argument("-iv", dest = "input_video", type = str, required = True, help = "Input video for demo.")
    parser.add_argument("-ot", dest = "od_threshold",  type = float, default = 0.3, help = "Detection Confidence threshold.")
    parser.add_argument("-ov", dest = "output_video", action = "store_true", help = "Input video for demo.")
    parser.add_argument("-s", dest = "skip_frame", type = int, default = 1, help = "Number of frame to skip.")
    parser.add_argument("-n", dest = "normalize_coor", action = "store_true", help = "Normalize GT anchor coordinate or not.")
    parser.add_argument("--mode", type = str, default = "only_final", choices = ["normal", "IPM", "only_final"], help = "Demo mode.")
    return parser.parse_args()

def demo_lane(net,  input_size, logger, draw, writer, skip_frame):
    wmin = 100
    net.target_available = False
    count = 0
    net.eval()
    # define drawer
    with torch.no_grad():
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                if count % skip_frame != 0:
                    count += 1
                    continue
                else:
                    count += 1
                # if count < 4000:
                    # continue
                print("count: {}".format(count))
                print("--------------------------")
                start = time.time()
                tic =time.time()
                # plt.cla()
                if draw.ax is not None:
                    draw.ax.clear()
                if draw.ax2 is not None:
                    draw.ax2.clear()
                if draw.ax3 is not None:
                    draw.ax3.clear()
                if draw.ax4 is not None:
                    draw.ax4.clear()
                if draw.ax5 is not None:
                    draw.ax5.clear()
                toc = time.time()
                print("Clear axes time: {:.3f}".format(toc - tic))

                tic = time.time()
                # resize image
                img = cv2.resize(frame, (input_size[1], input_size[0]), 0, 0, interpolation = cv2.INTER_LINEAR)
                img = img.astype(np.float32)
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
                scores, classification, transformed_anchors = net(input)
                toc =time.time()
                print("Net forward time: {}".format(toc - tic))

                tic =time.time()
                # image copy for drawing
                img_draw = img.copy()
                img_draw = img_draw.astype(np.uint8)
                toc =time.time()
                print("Pre-Post-process time: {}".format(toc - tic))                

                if draw.draw_local_maximum is not None:
                    draw.draw_local_maximum.imshow(img_draw)
                if draw.draw_clustering is not None:
                    draw.draw_clustering.imshow(img_draw)
                if draw.draw_poly_fit is not None:
                    draw.draw_poly_fit.imshow(img_draw)
                # draw.ax4.imshow(prob_map_seg, cmap = "gray")
                ##################
                # draw detection #
                ##################
                ipm_objs = [] if SHOW_IPM_OBJECT else None
                if SHOW_BBOX_RESULT:
                    tic = time.time()
                    idxs = np.where(scores > args.od_threshold) 
                    for j in range(idxs[0].shape[0]):
                        class_ =int(classification[idxs[0][j]])
                        bbox = transformed_anchors[idxs[0][j], :]
                        x1 = int(round(bbox[0]))
                        y1 = int(round(bbox[1]))
                        x2 = int(round(bbox[2]))
                        y2 = int(round(bbox[3]))

                        """ 
                        Create a Rectangle patch
                        Add the patch to Axes
                        """
                        if draw.draw_netout is not None:
                            rect = get_rect_patch(class_, x1, y1, x2, y2, LINE_WIDTH, dataset = dataset_type)
                            draw.draw_netout.add_patch(rect)
                        if draw.draw_local_maximum is not None:
                            rect = get_rect_patch(class_, x1, y1, x2, y2, LINE_WIDTH, dataset = dataset_type)
                            draw.draw_local_maximum.add_patch(rect)
                        if draw.draw_clustering is not None:
                            rect = get_rect_patch(class_, x1, y1, x2, y2, LINE_WIDTH, dataset = dataset_type)
                            draw.draw_clustering.add_patch(rect)
                        if draw.draw_poly_fit is not None:
                            rect = get_rect_patch(class_, x1, y1, x2, y2, LINE_WIDTH, dataset = dataset_type)
                            draw.draw_poly_fit.add_patch(rect)

                        if SHOW_IPM_OBJECT:
                            ipm_objs.append([x1, y2])
                            ipm_objs.append([x2, y2])
                        if DEBUG_MIN_OBJ_WIDTH:
                            if x2 - x1 < wmin:
                                wmin = x2 - x1
                    print("Min width: {}".format(wmin))

                # set limits and title
                height, width = img_draw.shape[0], img_draw.shape[1]
                tic = time.time()
                draw.set_limit_titles(height, width)
                toc = time.time()
                print("Set axes limit time: {}".format(toc - tic))
                end = time.time()
                if writer is not None:
                    writer.grab_frame()
                else:
                    plt.draw()
                    plt.waitforbuttonpress()
                    print("Total time: {}".format(end - start))
            else:
                break

if __name__ == "__main__":
    args = get_arguments()
    testing_log = "{}_{}x{}_demo_detection_{}".format(args.model, args.input_size[0], args.input_size[1], args.mode)
    logger = MyLog(testing_log)
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logger.info(testing_log)
    logger.info(args)

    # input video capture
    assert os.path.exists(args.input_video), "Input video {} does not exist.".format(args.input_video)
    cap = cv2.VideoCapture(args.input_video)
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
        output_path = os.path.join(output_dir, testing_log + "_" +args.input_video.split("/")[-1][:-4]+"_det.mp4")
        logger.info(output_path)
    # dataset
    # split = "val"
    # lib = importlib.import_module("dataset.bdd100kLaneandLineSeg")
    # test_dataset = lib.bdd100kLandandLineDataset(input_size = args.input_size, split = split)

    # network
    net = net_option(name = args.model, mode = "detection", normalize_anchor = args.normalize_coor)
    net = net.to(device)
    num_parameters = count_parameters(net)
    logger.info("Number of network parameters: {}".format(num_parameters))

    # resume from checkpoint
    assert os.path.exists(args.checkpoint), "Checkpoint {} does not exist.".format(args.checkpoint)
    state = torch.load(args.checkpoint)
    net.load_state_dict(state["model_state"],strict = True)
    logger.info("Resume from previous model {}".format(args.checkpoint))
    # demo
    draw = drawer(args.mode)
    if writer is not None:
        with writer.saving(draw.fig, output_path, 100):
            demo_lane(net, args.input_size, logger, draw, writer, args.skip_frame)
    else:
        demo_lane(net, args.input_size, logger, draw, writer, args.skip_frame)
    print(output_path)







