#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
from IPython.display import clear_output


# In[2]:


BUFFER_SIZE = 50000
BATCH_SIZE = 1
ALPHA = 1.0/10
GAMMA = 2.2
#MEAN=0.2
#STANDARD_DEVIATION=8


# In[3]:


encoder = tf.saved_model.load("encoder")
decoder = tf.saved_model.load("decoder")


# In[4]:


def load(image_file):#输入skybox(分辨率128*64)的绝对路径，返回一个32*128*3的图像
# Open the input file    
  file = OpenEXR.InputFile(image_file)
    
# Compute the size
  dw = file.header()['dataWindow']
  sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    

# Read the three color channels as 32-bit floats
  FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
  (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]
  #print(tf.shape(R))
# Normalize so that brightest sample is 1
  #brightest = max(R + G + B)
  #print("brightest",brightest)
  #R = [ i / brightest for i in R ]
  #G = [ i / brightest for i in G ]
  #B = [ i / brightest for i in B ]

  R =np.reshape(R,(sz[1],sz[0]))
  G =np.reshape(G,(sz[1],sz[0]))
  B =np.reshape(B,(sz[1],sz[0]))
  re = np.zeros((sz[1],sz[0],3),dtype=np.float32)  
  re[:,:,0]=R
  re[:,:,1]=G
  re[:,:,2]=B
  h=sz[1]//2
  re=re[:h,:,:]
  re[re<0]=0
  re=tf.constant(re,dtype=tf.float32)

  #hdrpath="C:/Users/A/Desktop/out.hdr"
  #imageio.imwrite(hdrpath,re,format='hdr')
  #print(re)
  return re


# In[5]:


def normalize(image):
    #image=ALPHA*(image**(1/GAMMA))-1
    image=ALPHA*(image**(1/GAMMA))
    
    #image=(image-trainDataMean)/trainDataStd
    return image
def deNormalize(image):    
    #image=((image+1)/ALPHA)**GAMMA
    image=(image/ALPHA)**GAMMA
    
    #image=image*trainDataStd+trainDataMean
    return image


# In[6]:


def generate_images_test(test_input, distorted_test_input,encoder,decoder):  
  normalized_test_input=normalize(test_input)

  encoder_output=encoder(normalized_test_input,training=False)
  encoder_output_distorted=encoder(normalize(distorted_test_input),training=False)

  decoder_output=decoder(encoder_output,training=False)
  decoder_output_distorted=decoder(encoder_output_distorted,training=False)

  display_list = [test_input[0,...], distorted_test_input[0,...], deNormalize(decoder_output[0,...]), deNormalize(decoder_output_distorted[0,...])]
  title = ['test_input', 'distorted_test_input', 'decoder_output','decoder_output_distorted']
  plt.figure(figsize=(20,20))
  #plt.figure()
  for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.title(title[i])    
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()


# In[7]:


from scipy.stats import truncnorm
def get_truncnorm(mean, sd, l, r):
    return truncnorm(
        (l - mean) / sd, (r - mean) / sd, loc=mean, scale=sd)
def truncated_normal(mean,sig,bound):
    return get_truncnorm(mean,sig,bound[0],bound[1]).rvs()
def truncated_lognormal(mean,sig,bound):
    x=get_truncnorm(mean,sig,math.log(bound[0]),math.log(bound[1])).rvs()
    return math.e**x
def radiometric_distortions(image):        
    image=np.array(image) 
    image[image < 0] = 0
    e=truncated_lognormal(0.2,math.sqrt(0.2),[0.1,10])
    #print("e:{0}".format(str(e)))
    image=e*image    
    lis=[]
    for i in range(3):
        wc=truncated_lognormal(0,0.06,[0.8,1.2])
        #print("wc:{0}".format(str(wc)))    
        lis.append(wc*image[...,i])    
    image=tf.stack(lis,axis=-1)    
    #print(image.shape)
    g=truncated_lognormal(0.0035,math.sqrt(0.2),[0.85,1.2])    
    image=image**(1.0/g)    
    #image=pow(image,0.5)
    #print("**:",tf.reduce_sum(tf.constant(image)))
    return image


# In[9]:

'''
batch_size=1
#加载测试集数据
PATH="E:/tensorflow/exrToExr/dataset/128X64/"
testImagePath=PATH+"test3500"

testDataset=tf.data.Dataset.list_files(testImagePath+'/*.exr')
testList=[]
for i in testDataset:
    tmp=load(i.numpy())
    testList.append(tmp);
maxV=-1e9    
minV=1e9       
for i in range(len(testList)):
#    v=normalize(testList[i])
    v=tf.constant(testList[i])
    maxV=max(tf.reduce_max(v),maxV)
    minV=min(tf.reduce_min(v),minV)
    
print("Max:",maxV)    
print("Min:",minV)            
testDataset=tf.data.Dataset.from_tensor_slices(testList)    
testDataset=testDataset.shuffle(BUFFER_SIZE)
print(testDataset)
testDataset=testDataset.batch(batch_size)
testDataNum=0
for i in testDataset:
    testDataNum+=1
print("testDataNum: ",testDataNum)   


# In[10]:


for inp in testDataset.take(100):
      generate_images_test(inp, radiometric_distortions(inp),encoder,decoder)

'''
# In[11]:


def get_encoder_output(path):#path: skybox的路径
    x=load(path)
    x=tf.expand_dims(x, 0)
    x=normalize(x)
    return encoder(x,training=False)


# In[12]:


def get_decoder_output(z):#未denormalize的output
    return decoder(z,training=False)


# In[15]:


def test():
    a=get_encoder_output("/Users/chenzeyin/Desktop/光照项目/dataSample-for-training/image2sky/target_test/000007skybox.exr")
    b=get_decoder_output(a)
    b=deNormalize(b[0])

# In[16]:


#test()


# In[ ]:




