import tensorflow as tf
import os
import OpenEXR
import Imath
import numpy as np
import glob
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import sys
import array
import OpenEXR
import Imath
import numpy as np
import imageio
import math
import random
import datetime
Mydevice = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(Mydevice[0], True)

BUFFER_SIZE = 50000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
ALPHA = 1.0 / 10
GAMMA = 2.2

def load(image_file):
    # Open the input file
    file = OpenEXR.InputFile(image_file)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
    # print(tf.shape(R))
    # Normalize so that brightest sample is 1
    # brightest = max(R + G + B)
    # print("brightest",brightest)
    # R = [ i / brightest for i in R ]
    # G = [ i / brightest for i in G ]
    # B = [ i / brightest for i in B ]

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
    # hdrpath="C:/Users/A/Desktop/out.hdr"
    # imageio.imwrite(hdrpath,re,format='hdr')
    # print(re)
    return re



def normalize(image):
    # image=ALPHA*(image**(1/GAMMA))-1
    image = ALPHA * (image ** (1 / GAMMA))

    # image=(image-trainDataMean)/trainDataStd
    return image


def deNormalize(image):
    # image=((image+1)/ALPHA)**GAMMA
    image = (image / ALPHA) ** GAMMA

    # image=image*trainDataStd+trainDataMean
    return image



encoder = tf.saved_model.load('./Model/Encoder')
decoder = tf.saved_model.load('./Model/Decoder')
input_dir = './testdirMask4000/'
target_dir = './testdir4000/'

input_path = glob.glob(input_dir +'*.exr')
input_path.sort()

target_path = glob.glob(target_dir+'*.exr')
target_path.sort()
start = time.time()
for i in range(1,30):
    j = np.random.rand()*5000
    j = int(j)
    input_img = load(input_path[i])
    target_img = load(target_path[i])
    r = (normalize(input_img))
    r = tf.expand_dims(r,0)
    r = encoder(r)
    r = deNormalize(decoder(r))
    s = str(j) +' pic'
    plt.title(s)
    plt.subplot(1,3,1)
    plt.imshow(input_img)
    plt.subplot(1,3,2)
    plt.imshow(target_img)
    plt.subplot(1,3,3)
    plt.imshow(r[0,...])
    plt.savefig('./ResultPic/GenPic/'+str(j)+'.jpg')
    print('Epoch {}, Time:{}'.format(
       i,time.time()-start
    )
    )

