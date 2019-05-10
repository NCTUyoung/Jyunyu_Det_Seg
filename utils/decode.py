
import cv2
import numpy as np
def decode(pred, cmap):
    """
    pred  : [batch, h, w]
    """
    pred  = pred[0]
    RGB = np.zeros((pred.shape[0], pred.shape[1], 3))
    for i in range(len(cmap)):
        RGB[pred == i] = cmap[i]
    RGB = RGB[:,:,[2,1,0]]
    # RGB = RGB.astype(np.uint8)
    return RGB