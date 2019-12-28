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
            wc = truncated_lognormal(0, 0.06, [0.8, 1.2])




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
    h = sz[1] // 2
    re = re[:h, :, :]
    re[re < 0] = 0

    re = tf.constant(re, dtype=tf.float32)
    re = tf.image.resize(re, [32, 128])

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

if __name__ == '__main__':
    src = '/media/czy/DataDisk/czy/40000data/skybox/020244skybox.exr'
    src_img = load(src)
    plt.title('source_EXR')
    plt.imshow(src_img)
    plt.show()
    for i in range(10):
        dist_img = radiometric_distortions(src_img)

        gray = np.random.uniform(1,2)
        dist_img = exr2png(gray,dist_img)
        plt.title('gray:'+str(gray))
        plt.imshow((dist_img.astype(int)))
        plt.show()


