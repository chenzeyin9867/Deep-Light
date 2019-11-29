# coding: utf-8

# In[1]:

from __future__ import absolute_import, division, print_function, unicode_literals
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
from IPython.display import clear_output
import glob
Mydevice = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(Mydevice[0], True)

# In[2]:


BUFFER_SIZE = 50000
BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256
ALPHA = 1.0 / 10
GAMMA = 2.2


# MEAN=0.2
# STANDARD_DEVIATION=8


# In[3]:


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


# In[4]:


from scipy.stats import truncnorm


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


# In[7]:

# modeify the batch size
batch_size = 8

# 训练集和测试集存的是没有归一化的数据
# 加载训练集数据
# modify by czy



testMaskDir='./testdirMask4000'
testDir = './testdir4000'
#
MaskPath = glob.glob(testMaskDir+'/*.exr')
MaskPath.sort()
Path = glob.glob(testDir+'/*.exr')
Path.sort()

testMaskDataset=tf.data.Dataset.from_tensor_slices(MaskPath)
testDataset = tf.data.Dataset.from_tensor_slices(Path)

testDataset = tf.data.Dataset.zip((testMaskDataset,testDataset))
testDataset=testDataset.shuffle(BUFFER_SIZE)
testDataset=testDataset.batch(batch_size)
testDataNum=0
for i in testDataset:
    testDataNum+=1
print("testDataNum: ",testDataNum)



# trainImagePath=PATH+"train_input"
trainImagePath = './trainMaskdir32000'
targetImagePath = "./traindir32000"
MaskPath = glob.glob(trainImagePath+'/*.exr')
MaskPath.sort()
Path = glob.glob(targetImagePath+'/*.exr')
Path.sort()

trainDataset=tf.data.Dataset.from_tensor_slices(MaskPath)
trainTargetDataset = tf.data.Dataset.from_tensor_slices(Path)
trainDataset = tf.data.Dataset.zip((trainDataset, trainTargetDataset))

trainDataset = trainDataset.shuffle(BUFFER_SIZE)
trainDataset = trainDataset.batch(batch_size)

trainDataNum = 0
for i, j in trainDataset:
    trainDataNum += i.shape[0]
print("trainDataNum: ", trainDataNum)






validationMask='./validateMaskdir4000'
validationDir = './validatedir4000'
#
MaskPath = glob.glob(testMaskDir+'/*.exr')
MaskPath.sort()
Path = glob.glob(testDir+'/*.exr')
Path.sort()

validateMaskDataset=tf.data.Dataset.from_tensor_slices(MaskPath)
validationDataset = tf.data.Dataset.from_tensor_slices(Path)
validationDataset = tf.data.Dataset.zip((validateMaskDataset,validationDataset))
validationDataset=validationDataset.shuffle(BUFFER_SIZE)
validationDataset=validationDataset.batch(batch_size)
validationDataNum=0
for i in validationDataset:
    validationDataNum+=1
print("validationDataNum: ",validationDataNum)


# In[12]:





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


# In[14]:

#
# maxV=-1e9
# minV=1e9
# for item in trainDataset.take(num):
#     x=tf.reduce_max(normalize(item))
#     y=tf.reduce_min(normalize(item))
#     maxV=max(maxV,x)
#     minV=min(minV,y)
# print(maxV)
# print(minV)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


# In[17]:


# In[18]:


from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


# Conv2d+BatchNorm2d+ReLU
def cbr(input_tensor, filters, kernel_size, strides):
    #  x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
    #                    use_bias=False, kernel_initializer='he_normal',
    #                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(input_tensor)

    x = layers.BatchNormalization(axis=-1,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
    x = layers.Activation('relu')(x)
    return x

# Conv2d+BatchNorm2d
def cbr_NoRelu(input_tensor, filters, kernel_size, strides):
    #  x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
    #                    use_bias=False, kernel_initializer='he_normal',
    #                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(input_tensor)

    x = layers.BatchNormalization(axis=-1,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)

    return x



def residual_1(input_tensor, filters):
    x = cbr(input_tensor, filters, 3, 2)
    #  y=layers.Conv2D(filters, 1, use_bias=False,
    #                    kernel_initializer='he_normal',
    #                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
    y = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_initializer='he_normal')(x)

    y = layers.BatchNormalization(axis=-1,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(y)
    y = cbr(y, filters, 1, 1)
    x = layers.add([x, y])
    x = cbr(x, filters, 1, 1)
    y = cbr(x, filters, 1, 1)
    x = layers.add([x, y])
    return x

def NewResidual_1(input_tensor,filters):
    x = input_tensor
    y = cbr(input_tensor,filters,3,2)
    y = cbr_NoRelu(y,filters,3,1)
    x = cbr_NoRelu(x,filters,1,2)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    x = y
    y = cbr(y,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    return y




def residual_2(input_tensor, filters):
    x = residual_1(input_tensor, filters)
    return x

def NewResidual_2(input_tensor,filters):
    x = NewResidual_1(input_tensor,filters)
    return x


def skyEncoder():
    inputs = tf.keras.layers.Input(shape=[32, 128, 3])
    initializer = tf.random_normal_initializer(0., 0.02)
    # result = tf.keras.Sequential()
    # result.add(
    #    tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
    #                           kernel_initializer=initializer, use_bias=False))
    result = tf.keras.layers.Conv2D(64, 7, strides=1, padding='same', activation=tf.nn.elu,
                                    kernel_initializer=initializer, use_bias=False)(inputs)
    # result = residual_1(result, 64)
    # result = residual_2(result, 16)
    result = NewResidual_1(result,64)
    result = NewResidual_2(result,16)
    result = tf.keras.layers.Flatten()(result)
    result = tf.keras.layers.Dense(64)(result);

    # result=result(inputs)
    return tf.keras.Model(inputs=inputs, outputs=result)


# In[ ]:


encoder = skyEncoder()
playDataset = trainDataset.take(1)






# In[21]:


def residual_3(input_tensor, filters):
    x = cbr(input_tensor, filters, 1, 1)
    #  y=layers.Conv2D(filters, 1, use_bias=False,
    #                    kernel_initializer='he_normal',
    #                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(x)
    y = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_initializer='he_normal')(x)

    y = layers.BatchNormalization(axis=-1,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(y)
    y = cbr(y, filters, 1, 1)
    x = layers.add([x, y])

    x = cbr(x, filters, 1, 1)
    y = cbr(x, filters, 1, 1)
    x = layers.add([x, y])

    x = cbr(x, filters, 1, 1)
    y = cbr(x, filters, 1, 1)
    x = layers.add([x, y])
    return x


def NewResidual_3(input_tensor,filters):
    #1 block
    x = input_tensor
    y = cbr(input_tensor,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    x = cbr_NoRelu(x,filters,1,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    #2 block
    x = y
    y = cbr(y, filters, 3, 1)
    y = cbr_NoRelu(y, filters, 3, 1)
    y = layers.add([x, y])
    y = layers.Activation('relu')(y)
    # 3 block
    x = y
    y = cbr(y, filters, 3, 1)
    y = cbr_NoRelu(y, filters, 3, 1)
    y = layers.add([x, y])
    y = layers.Activation('relu')(y)
    return y





def residual_4(input_tensor, filters):
    x = cbr(input_tensor, filters, 1, 1)
    y = cbr(x, filters, 1, 1)
    x = layers.add([x, y])

    x = cbr(input_tensor, filters, 1, 1)
    y = cbr(x, filters, 1, 1)
    x = layers.add([x, y])

    return x

def NewResidual_4(input_tensor,filters):
    x = input_tensor
    y = cbr(input_tensor,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    #2nd block
    x = y
    y = cbr(y,filters,3,1)
    y = cbr_NoRelu(y,filters,3,1)
    y = layers.add([x,y])
    y = layers.Activation('relu')(y)
    return y


def skyDecoder():
    inputs = tf.keras.layers.Input(shape=[64])
    result = tf.keras.layers.Dense(4096)(inputs)
    result = layers.Activation('elu')(result)
    result = tf.keras.layers.Reshape([8, 32, 16])(result)
    result = upsample(16, 3)(result)
    # result = residual_3(result, 64)
    result = NewResidual_3(result,64)
    result = upsample(64, 3)(result)
    result = NewResidual_4(result, 64)
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(
        result)
    return tf.keras.Model(inputs=inputs, outputs=result)


# In[22]:


decoder = skyDecoder()


LAMBDA = 100


# In[25]:


def loss_d(encoder_output, encoder_distorted_output):
    return tf.reduce_sum(tf.abs(encoder_output - encoder_distorted_output)) / encoder_output.shape[0]
    # return tf.reduce_sum(tf.abs(encoder_output - encoder_distorted_output))
    # return tf.abs(encoder_output - encoder_distorted_output)


# In[26]:


def loss_r(P, Pj, Pjd):
    # 反归一化后计算loss
    # P=deNormalize(P)
    # Pj=deNormalize(Pj)
    # Pjd=deNormalize(Pjd)
    return (tf.reduce_sum(tf.abs(Pj - P)) + tf.reduce_sum(tf.abs(Pjd - P))) / P.shape[0]
    # return tf.reduce_sum(tf.abs(Pj - P))+tf.reduce_sum(tf.abs(Pjd - P))
    # return tf.abs(Pj - P)+tf.abs(Pjd - P)


# In[27]:


def loss_s(P, Pj, Pjd, encoder_output, encoder_distorted_output):
    return loss_r(P, Pj, Pjd) + LAMBDA * loss_d(encoder_output, encoder_distorted_output)
    # return tf.reduce_sum(loss_r(P,Pj,Pjd))+LAMBDA*tf.reduce_sum(loss_d(encoder_output,encoder_distorted_output))


# In[28]:


# In[29]:


def loss_similarity(imageOrigin, imageOutput, imageOutput_distorted):
    imageOrigin = deNormalize(tf.where(imageOrigin > 0, imageOrigin, 0))
    imageOutput = deNormalize(tf.where(imageOutput > 0, imageOutput, 0))
    imageOutput_distorted = deNormalize(tf.where(imageOutput_distorted > 0, imageOutput_distorted, 0))
    # imageOrigin=deNormalize(imageOrigin)
    # imageOutput=deNormalize(imageOutput)
    # imageOutput_distorted=deNormalize(imageOutput_distorted)
    return (tf.reduce_sum(tf.abs(imageOrigin - imageOutput)) + tf.reduce_sum(
        tf.abs(imageOrigin - imageOutput_distorted))) / imageOrigin.shape[0]


# In[30]:


# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-3,
#    decay_steps=10000,
#    decay_rate=0.9,
#    staircase=True)
lr_schedule = tf.Variable(1e-3)
encoder_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.999)
decoder_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.999)

# In[31]:


checkpoint_dir = './40000data-checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder_optimizer=encoder_optimizer,
                                 decoder_optimizer=decoder_optimizer,
                                 encoder=encoder,
                                 decoder=decoder
                                 )
checkpoint.restore(tf.train.latest_checkpoint('./40000data-checkpoint'))

# In[32]:


# Define our metrics
train_ls_loss = tf.keras.metrics.Mean('train_ls_loss', dtype=tf.float32)
train_ld_loss = tf.keras.metrics.Mean('train_ld_loss', dtype=tf.float32)
train_lr_loss = tf.keras.metrics.Mean('train_lr_loss', dtype=tf.float32)
train_similarity = tf.keras.metrics.Mean('train_similarity', dtype=tf.float32)

validation_ls_loss = tf.keras.metrics.Mean('validation_ls_loss', dtype=tf.float32)
validation_ld_loss = tf.keras.metrics.Mean('validation_ld_loss', dtype=tf.float32)
validation_lr_loss = tf.keras.metrics.Mean('validation_lr_loss', dtype=tf.float32)
validation_similarity = tf.keras.metrics.Mean('validation_similarity', dtype=tf.float32)

test_ls_loss = tf.keras.metrics.Mean('test_ls_loss', dtype=tf.float32)
test_ld_loss = tf.keras.metrics.Mean('test_ld_loss', dtype=tf.float32)
test_lr_loss = tf.keras.metrics.Mean('test_lr_loss', dtype=tf.float32)
test_similarity = tf.keras.metrics.Mean('test_similarity', dtype=tf.float32)

# In[33]:


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './logs/gradient_tape/' + current_time + '/train_input'
validation_log_dir = './logs/gradient_tape/' + current_time + '/validation'
test_log_dir = './logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# In[34]:


def generate_images(test_input, distorted_test_input, input_target):
    normalized_test_input = normalize(test_input)

    encoder_output = encoder(normalized_test_input, training=False)
    encoder_output_distorted = encoder(normalize(distorted_test_input), training=False)

    decoder_output = decoder(encoder_output, training=False)
    decoder_output_distorted = decoder(encoder_output_distorted, training=False)

    display_list = [test_input[0, ...], distorted_test_input[0, ...], deNormalize(decoder_output[0, ...]),
                    deNormalize(decoder_output_distorted[0, ...]), input_target[0, ...]]
    title = ['test_input', 'distorted_test_input', 'decoder_output', 'decoder_output_distorted', 'origin_target']
    plt.figure(figsize=(30, 30))
    # plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig('./ResultPic/'+str(np.random.random())+'.jpg')

    # plt.show()


# In[35]:


@tf.function
def train_step(input_image, distorted_input_image, input_target):
    input_image = normalize(input_image)
    distorted_input_image = normalize(distorted_input_image)
    input_target = normalize(input_target)
    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
        encoder_output = encoder(input_image, training=True)
        # print(len(encoder_output.variables))
        encoder_output_distorted = encoder(distorted_input_image, training=True)
        # print(len(encoder_output_distorted.variables))

        decoder_output = decoder(encoder_output, training=True)
        decoder_output_distorted = decoder(encoder_output_distorted, training=True)

        ls = loss_s(input_target, decoder_output, decoder_output_distorted, encoder_output, encoder_output_distorted)
        ld = loss_d(encoder_output, encoder_output_distorted)
        lr = loss_r(input_target, decoder_output, decoder_output_distorted)
        ls_similarity = loss_similarity(input_image, decoder_output, decoder_output_distorted)

        encoder_gradients = encoder_tape.gradient(ls, encoder.trainable_variables)
        decoder_gradients = decoder_tape.gradient(lr, decoder.trainable_variables)

        encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
        decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))

        train_ls_loss(ls)
        train_ld_loss(ld)
        train_lr_loss(lr)
        train_similarity(ls_similarity)

        # train_ls_loss(tf.reduce_sum(ls))
        # train_ld_loss(tf.reduce_sum(ld))
        # train_lr_loss(tf.reduce_sum(lr))
        # train_similarity(tf.reduce_sum(ls_similarity))


# In[36]:


@tf.function
def validation_step(input_image, distorted_input_image,target):
    input_image = normalize(input_image)
    distorted_input_image = normalize(distorted_input_image)
    encoder_output = encoder(input_image, training=False)
    encoder_output_distorted = encoder(distorted_input_image, training=False)
    target = normalize(target)
    decoder_output = decoder(encoder_output, training=False)
    decoder_output_distorted = decoder(encoder_output_distorted, training=False)

    ls = loss_s(target, decoder_output, decoder_output_distorted, encoder_output, encoder_output_distorted)
    ld = loss_d(encoder_output, encoder_output_distorted)
    lr = loss_r(target, decoder_output, decoder_output_distorted)
    ls_similarity = loss_similarity(input_image, decoder_output, decoder_output_distorted)

    validation_ls_loss(ls)
    validation_ld_loss(ld)
    validation_lr_loss(lr)
    validation_similarity(ls_similarity)


# In[37]:


@tf.function
def test_step(input_image, distorted_input_image,target):
    input_image = normalize(input_image)
    target = normalize(target)
    distorted_input_image = normalize(distorted_input_image)
    encoder_output = encoder(input_image, training=False)
    encoder_output_distorted = encoder(distorted_input_image, training=False)

    decoder_output = decoder(encoder_output, training=False)
    decoder_output_distorted = decoder(encoder_output_distorted, training=False)

    ls = loss_s(target, decoder_output, decoder_output_distorted, encoder_output, encoder_output_distorted)
    ld = loss_d(encoder_output, encoder_output_distorted)
    lr = loss_r(target, decoder_output, decoder_output_distorted)
    ls_similarity = loss_similarity(input_image, decoder_output, decoder_output_distorted)

    test_ls_loss(ls)
    test_ld_loss(ld)
    test_lr_loss(lr)
    test_similarity(ls_similarity)


# In[38]:


def string2pic(str1,str2):

    pic1 = []
    pic2 = []
    for i in str1:
        a = load(i.numpy())
        pic1.append(a[tf.newaxis,...])


    for i in str2:
        pic2.append(load(i.numpy())[tf.newaxis,...])

    # pic1 = np.array(pic1)
    # pic2 = np.array(pic2)
    pic1 = tf.concat(pic1,0)
    pic2 = tf.concat(pic2,0)
    return pic1,pic2


def train(dataset, epochs):
    min_loss = 1e9
    accumulated_epoch_num = 0
    max_not_decrease_epoch_num = 10
    lr_decay_rate = 0.9
    for epoch in range(1, epochs):

        start = time.time()
        iteration = 0
        for input_image, input_target in dataset:

            iteration +=1
            input_image,input_target = string2pic(input_image,input_target)
            train_step(input_image, radiometric_distortions(input_image), input_target)
            if iteration % 200 ==0:
                print('Time taken for iteration {} is {} sec\n'.format(iteration, time.time() - start))
        # 将train loss输入到tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('ls_loss', train_ls_loss.result(), step=epoch)
            tf.summary.scalar('ld_loss', train_ld_loss.result(), step=epoch)
            tf.summary.scalar('lr_loss', train_lr_loss.result(), step=epoch)
            tf.summary.scalar('similarity', train_similarity.result(), step=epoch)


        for input_image,input_target in validationDataset:
          input_image,input_target = string2pic(input_image,input_target)
          validation_step(input_image, radiometric_distortions(input_image),input_target)
        #将validation loss输入到tensorboard
        with validation_summary_writer.as_default():
          tf.summary.scalar('ls_loss', validation_ls_loss.result(), step=epoch)
          tf.summary.scalar('ld_loss', validation_ld_loss.result(), step=epoch)
          tf.summary.scalar('lr_loss', validation_lr_loss.result(), step=epoch)
          tf.summary.scalar('similarity', validation_similarity.result(), step=epoch)

        for input_image,input_target in testDataset:
          input_image,input_target = string2pic(input_image,input_target)
          test_step(input_image, radiometric_distortions(input_image),input_target)
        #将test loss输入到tensorboard
        with test_summary_writer.as_default():
          tf.summary.scalar('ls_loss', test_ls_loss.result(), step=epoch)
          tf.summary.scalar('ld_loss', test_ld_loss.result(), step=epoch)
          tf.summary.scalar('lr_loss', test_lr_loss.result(), step=epoch)
          tf.summary.scalar('similarity', test_similarity.result(), step=epoch)

        clear_output(wait=True)
        for inp, j in testDataset.take(1):
            inp,j = string2pic(inp,j)

            generate_images(inp, radiometric_distortions(inp), j)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        #update learning rate,与validation的结果比较
        if min_loss>validation_ls_loss.result():
            min_loss=validation_ls_loss.result()
            accumulated_epoch_num=0
        else:
            accumulated_epoch_num+=1
            if accumulated_epoch_num>=max_not_decrease_epoch_num:
                accumulated_epoch_num=0
                #lr_schedule.assign(max(min_lr_rate,lr_schedule*lr_decay_rate))
                lr_schedule.assign(lr_schedule*lr_decay_rate)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        print('min_loss: ', min_loss)
        print('lr: ', lr_schedule)
        print(
            'Epoch {}, train_ls: {}, train_similarity: {}, validation_ls: {}, validation_similarity: {}, test_ls: {}, test_similarity: {}'.format(
                epoch + 1,
                train_ls_loss.result(),
                train_similarity.result(),
                validation_ls_loss.result(),
                validation_similarity.result(),
                test_ls_loss.result(),
                test_similarity.result()))
        # Reset metrics every epoch
        train_ls_loss.reset_states()
        train_ld_loss.reset_states()
        train_lr_loss.reset_states()
        train_similarity.reset_states()

        validation_ls_loss.reset_states()
        validation_ld_loss.reset_states()
        validation_lr_loss.reset_states()
        validation_similarity.reset_states()

        test_ls_loss.reset_states()
        test_ld_loss.reset_states()
        test_lr_loss.reset_states()
        test_similarity.reset_states()


# In[39]:


EPOCHS = 150

# In[ ]:


train(trainDataset, EPOCHS)
tf.saved_model.save(encoder, "./Model/Encoder")
tf.saved_model.save(decoder, "./Model/Decoder")


