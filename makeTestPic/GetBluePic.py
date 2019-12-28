import os

from scipy.stats import truncnorm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import loadModel
import tensorflow as tf
import math
import OpenEXR
import Imath
import exr2png
import array
def genSynthetic(exrPath,maskPath):

    #exrImage = cv2.imread(exrPath,cv2.IMREAD_UNCHANGED)
    maskImage = cv2.imread(maskPath)

    maskImage = transform.resize(maskImage, (64,128))

    source = cv2.imread(exrPath).astype('float32')

    down_size = 10

    new_region = maskImage == 0
    source[new_region] = 0
    return source

def cv2plt(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img

def get_truncnorm(mean, sd, l, r):
    return truncnorm(
        (l - mean) / sd, (r - mean) / sd, loc=mean, scale=sd)


def truncated_normal(mean, sig, bound):
    return get_truncnorm(mean, sig, bound[0], bound[1]).rvs()


def truncated_lognormal(mean, sig, bound):
    x = get_truncnorm(mean, sig, math.log(bound[0]), math.log(bound[1])).rvs()
    return math.e ** x


def radiometric_distortions(image):
    image = np.array(image)
    image[image < 0] = 0
    e = truncated_lognormal(0.2, math.sqrt(0.2), [0.1, 10])
    # print("e:{0}".format(str(e)))
    image = e * image
    b = 0
    lis = []
    for i in range(3):
        if i == 0: # R
            wc = truncated_lognormal(0, 0.06, [0.1,0.2])
        if i == 1: # G
            wc = truncated_lognormal(0, 0.06, [0.3,0.4])
        if i == 2: # B
            wc = truncated_lognormal(0, 0.06, [0.5, 2.0])




        # print("wc:{0}".format(str(wc)))
        lis.append(wc * image[..., i])
    image = tf.stack(lis, axis=-1)
    # print(image.shape)
    g = truncated_lognormal(0.0035, math.sqrt(0.2), [0.85, 1.2])
    image = image ** (1.0 / g)
    # image=pow(image,0.5)
    # print("**:",tf.reduce_sum(tf.constant(image)))
    return image


def load(image_file):

    file = OpenEXR.InputFile(image_file)

    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]

    R = np.reshape(R, (sz[1], sz[0]))
    G = np.reshape(G, (sz[1], sz[0]))
    B = np.reshape(B, (sz[1], sz[0]))
    re = np.zeros((sz[1], sz[0], 3), dtype=np.float32)
    re[:, :, 0] = R
    re[:, :, 1] = G
    re[:, :, 2] = B

    re[re < 0] = 0

    re = tf.constant(re, dtype=tf.float32)
    re = tf.image.resize(re, [64, 128])

    return re



def exr2png(gray,img):

    # print(img)
    # cv2.imshow('1', img)
    img = img.numpy()
    img[img < 0] = 0
    delt = 0.001
    # print(img.shape)
    sum = img.shape[0] * img.shape[1]
    mid = np.log(delt + img)
    # print(mid)
    logValue = mid.sum() / (3 * sum)
    final_v = np.exp(logValue)
    # print(final_v)


    Lxy = gray / final_v * img
    # print(Lxy)
    L = Lxy / (1. + Lxy)
    # print(L)
    # cv2.imshow('2', L)
    L = L * 255
    # L= L[0:32,:,:]
    # print(L.shape)

    return L


from PIL import Image
from PIL import ImageEnhance
def enhance(src):
    img = Image.open(src)
    # 对比度增强
    enh_con = ImageEnhance.Contrast(img)
    contrast = 1.5
    img_contrasted = enh_con.enhance(contrast)
    img_contrasted.show()
    return img_contrasted

import cv2
import matplotlib
from skimage import transform,data
import matplotlib.pyplot as plt
import os

import numpy as np

def genSynthetic(img,maskPath,gray):

    #exrImage = cv2.imread(exrPath,cv2.IMREAD_UNCHANGED)
    maskImage = cv2.imread(maskPath)

    maskImage = transform.resize(maskImage, (64,128))

    newPng = exr2png(gray,img)
    newPng = newPng.astype(int)
    down_size = 10
    new_region = np.ones((64,128,3))
    new_region[down_size:,...] = maskImage[0:64-down_size,...]
    new_region = new_region == 0
    newPng[new_region] = 0
    return newPng



import glob
if __name__ == '__main__':
    mask_dir = '/media/czy/DataDisk/czy/40000data/mask'
    skybox_dir = '/media/czy/DataDisk/czy/40000data/skybox'
    mask_files = os.listdir(mask_dir)
    skybox_files = os.listdir(skybox_dir)
    mask_files.sort()
    skybox_files.sort()
    SynDir= '/media/czy/DataDisk/czy/newDeepBlue2019.12.28/Syn'
    SkyboxSaveDir = '/media/czy/DataDisk/czy/newDeepBlue2019.12.28/skybox/'
    index = 2000000
    for skybox_path in skybox_files:
        #get the skybox
        skybox = load(skybox_dir +'/'+skybox_path)
        for i in range(2):
            radio_skybox = radiometric_distortions(skybox)
            rnd = np.random.randint(1,40000,1)

            gray = np.random.uniform(0.1,0.4)
            radio_skybox_png = exr2png(gray,radio_skybox)
            synthetic_pic = genSynthetic(radio_skybox,mask_dir+'/'+mask_files[rnd[0]],gray)
            index +=1
            cv2.imwrite(SynDir+'/'+str(index)+'.png',cv2plt(synthetic_pic))
            cv2.imwrite(SkyboxSaveDir+'/' +str(index)+'.png',cv2plt(radio_skybox_png))
            print(index)





