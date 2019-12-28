import cv2
import matplotlib
import exr2png
from skimage import transform,data
import matplotlib.pyplot as plt
import os

import numpy as np

def genSynthetic(exrPath,maskPath,gray):

    #exrImage = cv2.imread(exrPath,cv2.IMREAD_UNCHANGED)
    maskImage = cv2.imread(maskPath)

    maskImage = transform.resize(maskImage, (64,128))

    newPng = exr2png.exr2png(gray,exrPath)
    newPng = newPng.astype(int)
    down_size = 10
    new_region = np.ones((64,128,3))
    new_region[down_size:,...] = maskImage[0:64-down_size,...]
    new_region = new_region == 0
    newPng[new_region] = 0
    return newPng




