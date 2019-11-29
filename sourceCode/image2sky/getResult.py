#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 指定训练使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import random
from PIL import Image
import OpenEXR
import sourceCode.image2sky.load_model as load_model
import array

import sourceCode.image2sky.densenet as densenet



# In[2]:


INPUT_IMG_WIDTH = 224
INPUT_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 128
TARGET_IMG_HEIGHT = 32
LEARNING_RATE1 = 3e-4
LEARNING_RATE2 = 2e-6
BETA1 = 0.4
BETA2 = 0.999


# In[3]:


def load_img(image_file):
    img = Image.open(image_file)
    img = img.resize((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH))
    img = tf.constant(np.asarray(img), dtype=tf.float32)
    
    return img


# In[4]:


def show(imgPath, exrPath, savePath, exrOutputPath):
    img = load_img(imgPath)
    exr = load_model.load(exrPath)
    
    target_z = load_model.get_encoder_output(exrPath)
    recon_img = load_model.get_decoder_output(target_z)
    recon_img = load_model.deNormalize(recon_img[0])
    
    plt.figure(figsize=(20, 14))
    plt.subplot(4, 1, 1)
    plt.title('input_image')
    plt.imshow(img.numpy().astype(int))
    plt.subplot(4, 1, 2)
    plt.title('target_image')
    plt.imshow(exr.numpy())
    plt.subplot(4, 1, 3)
    plt.title('reconstruct_image')
    plt.imshow(recon_img.numpy())
    plt.subplot(4, 1, 4)
    res_z = denseNet_model(tf.expand_dims((img - 128) / 255.0, axis=0))
    res_img = load_model.get_decoder_output(res_z)
    res_img = res_img[0]
    res_img = np.where(res_img > 0, res_img, 0)
    res_img = load_model.deNormalize(res_img)
    plt.title('result_image')
    plt.imshow(res_img)
    
    plt.savefig(savePath)
    
    plt.close()
    
    print(savePath + '  saved.')


def getResult(imgPath, exrOutputPath):
    img = load_img(imgPath)
    res_z = denseNet_model(tf.expand_dims((img - 128) / 255.0, axis=0))
    res_img = load_model.get_decoder_output(res_z)
    res_img = res_img[0]
    res_img = np.where(res_img > 0, res_img, 0)
    res_img = load_model.deNormalize(res_img)
    
    padding = np.zeros([32, 128, 3])
    res_img_new = np.concatenate((res_img, padding))
    
    exr_res = OpenEXR.OutputFile(exrOutputPath, OpenEXR.Header(128, 64))
    dataR = array.array('f', res_img_new[:, :, 0].flatten()).tobytes()
    dataG = array.array('f', res_img_new[:, :, 1].flatten()).tobytes()
    dataB = array.array('f', res_img_new[:, :, 2].flatten()).tobytes()
    exr_res.writePixels({'R':dataR, 'G':dataG, 'B':dataB})


# In[5]:


denseNet_model = densenet.DenseNet(10, 48, 4, 64, [6, 12, 36, 24], "channels_last",
                                weight_decay=1e-7, pool_initial=True)


# In[6]:


lr = tf.Variable(LEARNING_RATE1)
image2sky_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, 
                                                beta_1=BETA1, beta_2=BETA2)
# 自定义load哪一个checkpoint的数据
checkpoint_dir = "./image2skyModel/ckpt-24"

checkpoint = tf.train.Checkpoint(image2sky_optimizer=image2sky_optimizer,
                                    denseNet_model=denseNet_model)

checkpoint.restore(checkpoint_dir)
lr.assign(LEARNING_RATE1)


# In[7]:


def getResultAndShow():
    PATH_INPUT = '../../dataSample-for-getResult/image2sky/input'
    PATH_TARGET = '../../dataSample-for-getResult/image2sky/target'
    test_exrs = os.listdir(PATH_TARGET)
    test_exrs.sort()
    test_img_dirs = os.listdir(PATH_INPUT)
    test_img_dirs.sort()

    fileNum = len(test_exrs)

    for i in range(fileNum):
        exr_img_path = os.path.join(PATH_TARGET, test_exrs[i]).replace("\\", "/")
        crop_dir_path = os.path.join(PATH_INPUT, test_img_dirs[i]).replace("\\", "/")
        crop_imgs = os.listdir(crop_dir_path)
        crop_imgs.sort()

        for j in range(len(crop_imgs)):
            crop_img_path = os.path.join(crop_dir_path, crop_imgs[j]).replace("\\", "/")

            savePath = './result_show/'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            savePath = savePath + test_img_dirs[i]
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            savePath = savePath + '/crop' + str(j) + '.png'

            exrOutputPath = './result_exr/'
            if not os.path.exists(exrOutputPath):
                os.mkdir(exrOutputPath)
            exrOutputPath = exrOutputPath + test_img_dirs[i]
            if not os.path.exists(exrOutputPath):
                os.mkdir(exrOutputPath)
            exrOutputPath = exrOutputPath + '/crop' + str(j) + '.exr'

            getResult(crop_img_path, exrOutputPath)
            show(crop_img_path, exr_img_path, savePath, exrOutputPath)


# In[8]:


def getResultOnly():
    PATH_INPUT = '../../dataSample-for-getResult/image2sky/input'
    test_img_dirs = os.listdir(PATH_INPUT)
    
    fileNum = len(test_img_dirs)
    
    for i in range(fileNum):
        crop_dir_path = os.path.join(PATH_INPUT, test_img_dirs[i]).replace("\\", "/")
        crop_imgs = os.listdir(crop_dir_path)
        crop_imgs.sort()
        
        for j in range(len(crop_imgs)):
            crop_img_path = os.path.join(crop_dir_path, crop_imgs[j]).replace("\\", "/")

            exrOutputPath = './result_exr/'
            if not os.path.exists(exrOutputPath):
                os.mkdir(exrOutputPath)
            exrOutputPath = exrOutputPath + test_img_dirs[i]
            if not os.path.exists(exrOutputPath):
                os.mkdir(exrOutputPath)
            exrOutputPath = exrOutputPath + '/crop' + str(j) + '.exr'

            getResult(crop_img_path, exrOutputPath)


# In[9]:


getResultAndShow()
print("Done.")

