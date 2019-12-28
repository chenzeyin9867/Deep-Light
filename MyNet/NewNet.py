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
from PIL import Image
import cv2

Mydevice = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(Mydevice[0], True)

# In[2]:sad


BUFFER_SIZE = 50000
BATCH_SIZE = 64
IMG_WIDTH = 256
IMG_HEIGHT = 256
ALPHA = 1.0 / 10
GAMMA = 2.2




batch_size = 64

# 训练集和测试集存的是没有归一化的数据
# 加载训练集数据
# modify by czy



testMaskDir='/home/czy/DataSet/bluerpic20191227/testdirMask4000'
testDir = '/home/czy/DataSet/bluerpic20191227/testdir4000'
#
MaskPath = glob.glob(testMaskDir+'/*.png')
MaskPath.sort()

Path = glob.glob(testDir+'/*.png')
Path.sort()
import random
a = list(range(10000))
b = random.sample(a,100)
genTestMask = [MaskPath[i] for i in b]
genSkybox = [Path[i] for i in b]




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
trainImagePath = '/home/czy/DataSet/bluerpic20191227/trainMaskdir32000'
targetImagePath = "/home/czy/DataSet/bluerpic20191227/traindir32000"
MaskPath = glob.glob(trainImagePath+'/*.png')
MaskPath.sort()

Path = glob.glob(targetImagePath+'/*.png')
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






validationMask='/home/czy/DataSet/bluerpic20191227/validateMaskdir4000'
validationDir = '/home/czy/DataSet/bluerpic20191227/validatedir4000'
#
MaskPath = glob.glob(testMaskDir+'/*.png')
MaskPath.sort()
Path = glob.glob(testDir+'/*.png')
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
    # image = ALPHA * (image ** (1 / GAMMA))
    image = (image - 127.5) / 127.5
    # image=(image-trainDataMean)/trainDataStd
    return image


def deNormalize(image):
    # image=((image+1)/ALPHA)**GAMMA
    # image = (image / ALPHA) ** GAMMA
    image = image * 127.5 + 127.5
    # image=image*trainDataStd+trainDataMean
    return image



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



def loss_r(P, Pj):
    # 反归一化后计算loss
    # P=deNormalize(P)
    # Pj=deNormalize(Pj)
    # Pjd=deNormalize(Pjd)
    return (tf.reduce_sum(tf.abs(Pj - P))) / P.shape[0]
    # return tf.reduce_sum(tf.abs(Pj - P))+tf.reduce_sum(tf.abs(Pjd - P))
    # return tf.abs(Pj - P)+tf.abs(Pjd - P)


lr_schedule = tf.Variable(1e-3)
encoder_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.999)
decoder_optimizer = tf.keras.optimizers.Adam(lr_schedule, 0.5, 0.999)

# In[31]:


checkpoint_dir = './checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(encoder_optimizer=encoder_optimizer,
                                 decoder_optimizer=decoder_optimizer,
                                 encoder=encoder,
                                 decoder=decoder
                                 )
checkpoint.restore(tf.train.latest_checkpoint('./checkpoint'))
print("The Latest ckpt:{}".format(tf.train.latest_checkpoint('./checkpoint')))
# In[32]:


# Define our metrics

train_lr_loss = tf.keras.metrics.Mean('train_lr_loss', dtype=tf.float32)
validation_lr_loss = tf.keras.metrics.Mean('validation_lr_loss', dtype=tf.float32)
test_lr_loss = tf.keras.metrics.Mean('test_lr_loss', dtype=tf.float32)


# In[33]:


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './logs/gradient_tape/' + current_time + '/train_input'
validation_log_dir = './logs/gradient_tape/' + current_time + '/validation'
test_log_dir = './logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# In[34]:
index = 1


def genpic(test_input,input_target,epoch):
    pic_dir = './Result/' + str(epoch)
    global index
    normalized_test_input = normalize(test_input)

    encoder_output = encoder(normalized_test_input, training=False)

    decoder_output = decoder(encoder_output, training=False)


    title = ['test_input', 'decoder_output', 'origin_target']
    if os.path.exists(pic_dir)==False:
      os.makedirs(pic_dir)
    display_list = [test_input, deNormalize(decoder_output),
                     input_target]

    for i in range(len(test_input)):
        plt.figure(figsize=(30, 30))
        for j in range(3):
            plt.subplot(1,3,j+1)
            plt.title(title[j])
            img = display_list[j][i].numpy().astype(int)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            plt.imshow(img)
            plt.axis('off')
        plt.savefig(pic_dir + '/' + str(index)+'.png')
        plt.close()
        index += 1




def generate_images(test_input, input_target):
    global  index
    normalized_test_input = normalize(test_input)

    encoder_output = encoder(normalized_test_input, training=False)


    decoder_output = decoder(encoder_output, training=False)


    display_list = [test_input[0, ...], deNormalize(decoder_output[0, ...]),
                     input_target[0, ...]]
    title = ['test_input',  'decoder_output',  'origin_target']
    plt.figure(figsize=(30, 30))
    # plt.figure()
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        img = display_list[i].numpy().astype(int)

        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        plt.imshow(img)

        plt.axis('off')
    plt.show()
    plt.savefig('./ResultPic/'+str(index)+'.jpg')
    index+=1
    # plt.show()


# In[35]:


@tf.function
def train_step(input_image, input_target):
    input_image = normalize(input_image)

    input_target = normalize(input_target)
    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
        encoder_output = encoder(input_image, training=True)

        decoder_output = decoder(encoder_output, training=True)

        lr = loss_r(input_target, decoder_output) #另一部分是该 decoder 对z的结果P̂和z d 的结果P̂ d 以及原图 Normalize 后的结果P之间的 loss(L r =‖P̂ − P‖ 1 + ‖P̂ d − P‖ 1 )。

        encoder_gradients = encoder_tape.gradient(lr, encoder.trainable_variables)
        decoder_gradients = decoder_tape.gradient(lr, decoder.trainable_variables)

        encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
        decoder_optimizer.apply_gradients(zip(decoder_gradients, decoder.trainable_variables))

        train_lr_loss(lr)

@tf.function
def validation_step(input_image,target):
    input_image = normalize(input_image)

    encoder_output = encoder(input_image, training=False)

    target = normalize(target)
    decoder_output = decoder(encoder_output, training=False)


    lr = loss_r(target, decoder_output)

    validation_lr_loss(lr)



# In[37]:


@tf.function
def test_step(input_image,target):
    input_image = normalize(input_image)
    target = normalize(target)

    encoder_output = encoder(input_image, training=False)

    decoder_output = decoder(encoder_output, training=False)


    lr = loss_r(target, decoder_output)


    test_lr_loss(lr)



# In[38]:

def getpic(str1,str2):
    pic1 = []
    pic2 = []
    for i in str1:
        a = cv2.imread(i)[:32, ...].astype('float32')
        pic1.append(a[tf.newaxis, ...])

    for i in str2:
        b = cv2.imread(i)[:32, ...].astype('float32')
        pic2.append(b[tf.newaxis, ...])

    # pic1 = np.array(pic1)
    # pic2 = np.array(pic2)
    pic1 = tf.concat(pic1, 0)
    pic2 = tf.concat(pic2, 0)
    return pic1, pic2

def string2pic(str1,str2):

    pic1 = []
    pic2 = []
    for i in str1:
        a = cv2.imread(i.numpy().decode())[:32,...].astype('float32')
        pic1.append(a[tf.newaxis,...])


    for i in str2:
        b = cv2.imread(i.numpy().decode())[:32,...].astype('float32')
        pic2.append(b[tf.newaxis,...])

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
            train_step(input_image, input_target)
            if iteration % 2000 ==0:
                print('Time taken for iteration {} is {} sec\n'.format(iteration, time.time() - start))
        # 将train loss输入到tensorboard
        with train_summary_writer.as_default():

            tf.summary.scalar('lr_loss', train_lr_loss.result(), step=epoch)



        for input_image,input_target in validationDataset:
          input_image,input_target = string2pic(input_image,input_target)
          validation_step(input_image,input_target)
        #将validation loss输入到tensorboard
        with validation_summary_writer.as_default():

          tf.summary.scalar('lr_loss', validation_lr_loss.result(), step=epoch)


        for input_image,input_target in testDataset:
          input_image,input_target = string2pic(input_image,input_target)
          test_step(input_image,input_target)
        #将test loss输入到tensorboard
        with test_summary_writer.as_default():

          tf.summary.scalar('lr_loss', test_lr_loss.result(), step=epoch)


        clear_output(wait=True)

        inp,j = getpic(genTestMask,genSkybox)

        genpic(inp,j,epoch)

        # saving (checkpoint) the model every 20 epochs

        checkpoint.save(file_prefix=checkpoint_prefix)

        # update learning rate,与validation的结果比较
        if min_loss>validation_lr_loss.result():
            min_loss=validation_lr_loss.result()
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
            'Epoch {}, train_lr: {}, validation_lr: {}, , test_lr: {}'.format(
                epoch + 1,
                train_lr_loss.result(),
                validation_lr_loss.result(),
                test_lr_loss.result()))
        # Reset metrics every epoch
        train_lr_loss.reset_states()
        validation_lr_loss.reset_states()
        test_lr_loss.reset_states()



# In[39]:


EPOCHS = 10

# In[ ]:


train(trainDataset, EPOCHS)
tf.saved_model.save(encoder, "./Model/Encoder")
tf.saved_model.save(decoder, "./Model/Decoder")


