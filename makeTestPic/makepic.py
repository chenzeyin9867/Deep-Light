import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import loadModel
import tensorflow as tf
def genSynthetic(exrPath,maskPath):

    #exrImage = cv2.imread(exrPath,cv2.IMREAD_UNCHANGED)
    maskImage = cv2.imread(maskPath)

    maskImage = transform.resize(maskImage, (64,128))

    source = cv2.imread(exrPath).astype('float32')

    down_size = 10
    down_size = 10

    new_region = maskImage == 0

    source[new_region] = 0
    return source

def cv2plt(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img






if __name__ == '__main__':

    source = '/home/czy/MyWorkSpace/Deep-Light/makeTestPic/imageSource/pano_agfudjtrbkpcoa.jpg'
    # mask ='/home/czy/MyWorkSpace/Deep-Light/makeTestPic/imageMask/pano_abnslvsfyaphuy.png'
    mask = '/home/czy/MyWorkSpace/Deep-Light/makeTestPic/imageMask'+'/'+os.path.basename(source).split('.')[0]+'.png'
    syn = genSynthetic(source,mask)
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
