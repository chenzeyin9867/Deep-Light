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
    lis = []
    for i in range(3):
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





if __name__ == '__main__':

    source = '/home/czy/testdirMask4000/174.png'
    syn = cv2.imread(source)
    source_img = syn
    Encoder = loadModel.skyEncoder()
    Decoder = loadModel.skyDecoder()

    checkpoint = tf.train.Checkpoint(
                                     encoder=Encoder,
                                     decoder=Decoder
                                     )
    checkpoint.restore(tf.train.latest_checkpoint('/home/czy/MyWorkSpace/Deep-Light/MyNet/checkpoint'))
    print(tf.train.latest_checkpoint('/home/czy/MyWorkSpace/Deep-Light/MyNet/checkpoint'))
    syn = syn[0:32,...]
    syn = syn[tf.newaxis,...]
    syn = loadModel.normalize(syn)
    encoder_out = Encoder(syn,training=False)
    decoder_out = Decoder(encoder_out,training = False)
    out = loadModel.deNormalize(decoder_out)


    img = cv2plt(source_img)[0:32,...]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img.astype(int))
    plt.subplot(1,2,2)
    img = out.numpy().astype(int)[0,...]
    img = cv2plt(img)
    plt.imshow(img)
    plt.show()
