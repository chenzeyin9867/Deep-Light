#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#
# # 指定训练使用的GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
# Mydevice = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(Mydevice[0], True)
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import load_model
import random
import densenet
import glob
from PIL import Image

# In[2]:


BUFFER_SIZE = 100
BATCH_SIZE = 16
INPUT_IMG_WIDTH = 224
INPUT_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 128
TARGET_IMG_HEIGHT = 32
PATH_TRAIN_INPUT_IMAGE = "./new"
PATH_TRAIN_TARGET_IMAGE = "./newSky"
PATH_TEST_INPUT_IMAGE = "../../dataSample-for-training/image2sky/input_test"
PATH_TEST_TARGET_IMAGE = "../../dataSample-for-training/image2sky/target_test"
LEARNING_RATE1 = 3e-4
LEARNING_RATE2 = 2e-6
BETA1 = 0.4
BETA2 = 0.999
IMAGE_2_SKY_FIR_EPOCH = 5
IMAGE_2_SKY_SEC_EPOCH = 55
IMAGE_2_AZIMUTH_EPOCH = 3
LAMBDA_C = 1e-3


# In[3]:




# 载入图片
def load_image_train(image_file):
    img = Image.open(image_file)

    img = img.resize((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH))
    img = tf.constant(np.asarray(img), dtype=tf.float32)

    return img


# 计算loss
# some question
def get_image2sky_loss(target_z, output_z):
    lossZ = tf.square(target_z - output_z)
    lossZ = tf.reduce_sum(lossZ)
    lossZ = lossZ / target_z.shape[0]



    lossC = load_model.get_decoder_output(target_z) - load_model.get_decoder_output(output_z)
    lossC = tf.reduce_sum(tf.abs(lossC))/target_z.shape[0]

    return lossZ, lossC, lossZ + LAMBDA_C * lossC


# In[6]:


# 定义loss矩阵，用来累计一个epoch内每个batch的loss和
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

loss_z = tf.keras.metrics.Mean('loss_z', dtype=tf.float32)
loss_c = tf.keras.metrics.Mean('loss_c', dtype=tf.float32)

# In[ ]:



@tf.function
def train_step(input_image, target_z):
    with tf.GradientTape() as image2sky_tape:
        output_z = denseNet_model(input_image)
        lossZ, lossC, image2sky_loss = get_image2sky_loss(target_z, output_z)

        train_loss(image2sky_loss)
        loss_z(lossZ)
        loss_c(lossC)

        image2sky_gradients = image2sky_tape.gradient(image2sky_loss,
                                                      denseNet_model.trainable_variables)
        image2sky_optimizer.apply_gradients(zip(image2sky_gradients,
                                                denseNet_model.trainable_variables))


# In[9]:


@tf.function
def test_step(input_image, target_z):
    output_z = denseNet_model(input_image)
    lossZ, lossC, image2sky_loss = get_image2sky_loss(target_z, output_z)

    test_loss(image2sky_loss)


# In[10]:


def train(train_dataset, test_dataset, epochs, checkpoint, checkpoint_prefix):
    input_image = []
    target_z = []
    for epoch in range(0, epochs):
        start = time.time()
        ite = 0
        for input_image_path, target_exr_path in train_dataset:
            input_image.clear()
            target_z.clear()
            ite+=1
            for i in range(len(input_image_path)):

                input_image.append((load_image_train(input_image_path[i].numpy().decode()) - 128) / 255.0)
                target_z.append(tf.squeeze(load_model.get_encoder_output(target_exr_path[i].numpy().decode())).numpy())


            input_image_tf = tf.stack(input_image)
            target_z_tf = tf.stack(target_z)
            #print("step {}".format(ite))
            train_step(input_image_tf, target_z_tf)

        # for input_image_path, target_z in test_dataset:
        #     input_image.clear()
        #     for i in range(BATCH_SIZE):
        #         input_image.append(load_image_train(input_image_path[i].numpy().decode()) - 128 / 255)
        #     input_image_tf = tf.stack(input_image)

            #test_step(input_image_tf, target_z)

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('loss_z', loss_z.result(), step=epoch)
            tf.summary.scalar('loss_c', loss_c.result(), step=epoch)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
        print('TrainLoss is {},  loss_z is {} ,loss_c is {}'.format(train_loss.result(),loss_z.result(),loss_c.result()))

        train_loss.reset_states()
        test_loss.reset_states()
        loss_z.reset_states()
        loss_c.reset_states()

        # if epoch % 5 == 0:
        #     show()
        if epoch %3 ==0:
            show()
        if epoch % 10 == 0:

            checkpoint.save(file_prefix=checkpoint_prefix)


# In[11]:


# 数据集初始化
train_dataset_input_image = []
train_dataset_target_z = []

test_dataset_input_image = []
test_dataset_target_z = []

train_input_image_dirs = os.listdir(PATH_TRAIN_INPUT_IMAGE)
train_input_image_dirs.sort()

test_input_image_dirs = os.listdir(PATH_TEST_INPUT_IMAGE)
test_input_image_dirs.sort()

train_target_images = os.listdir(PATH_TRAIN_TARGET_IMAGE)
train_target_images.sort()

test_target_images = os.listdir(PATH_TEST_TARGET_IMAGE)
test_target_images.sort()

# In[12]:


# 加载训练集数据
for i in range(len(train_input_image_dirs)):
    train_input_images_path = os.path.join(PATH_TRAIN_INPUT_IMAGE, train_input_image_dirs[i])
    train_input_images = os.listdir(train_input_images_path)
    train_input_images.sort()

    for train_image in train_input_images:
        train_image_path = os.path.join(train_input_images_path, train_image).replace("\\", "/")
        train_dataset_input_image.append(tf.constant(train_image_path))
    for j in range(len(train_input_images)):
        train_target_image_path = os.path.join(PATH_TRAIN_TARGET_IMAGE, train_target_images[i]).replace("\\", "/")
        train_target_true_image = glob.glob(train_target_image_path+'/*.exr')

        train_dataset_target_z.append(train_target_true_image[0])

# In[13]:



# 加载测试集数据
# for i in range(len(test_input_image_dirs)):
#     test_input_images_path = os.path.join(PATH_TEST_INPUT_IMAGE, test_input_image_dirs[i])
#     test_input_images = os.listdir(test_input_images_path)
#     test_input_images.sort()
#
#     for test_image in test_input_images:
#         test_image_path = os.path.join(test_input_images_path, test_image).replace("\\", "/")
#         test_dataset_input_image.append(tf.constant(test_image_path))
#     for j in range(len(test_input_images)):
#         test_target_image_path = os.path.join(PATH_TEST_TARGET_IMAGE, test_target_images[i]).replace("\\", "/")
#         test_dataset_target_z.append(tf.squeeze(load_model.get_encoder_output(test_target_image_path)))

# In[14]:


# 训练集数据input_image, target_z打包
train_dataset_input_image = tf.data.Dataset.from_tensor_slices(train_dataset_input_image)
train_dataset_target_z = tf.data.Dataset.from_tensor_slices(train_dataset_target_z)

train_dataset = tf.data.Dataset.zip((train_dataset_input_image, train_dataset_target_z))

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# In[15]:



# 测试集数据input_image, target_z打包
# test_dataset_input_image = tf.data.Dataset.from_tensor_slices(test_dataset_input_image)
# test_dataset_target_z = tf.data.Dataset.from_tensor_slices(test_dataset_target_z)
#
# test_dataset = tf.data.Dataset.zip((test_dataset_input_image, test_dataset_target_z))
#
# test_dataset = test_dataset.shuffle(BUFFER_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# In[16]:


# 定义模型对象
denseNet_model = densenet.DenseNet(10, 48, 4, 64, [6, 12, 36, 24], "channels_last",
                                   weight_decay=1e-7, pool_initial=True)

# In[17]:

def show():
    print('Train Dataset Result:')
    for img ,tar in train_dataset.take(1):

        print(' ')
        for i in range(1):
            path = img[i].numpy().decode()

            src = load_image_train(img[i].numpy().decode())
            dst = load_model.load(tar[i].numpy().decode())



            plt.figure(figsize=(20, 14))
            plt.subplot(3, 1, 1)
            plt.title((path))
            plt.imshow(src.numpy().astype(np.uint8))
            plt.subplot(3, 1, 2)
            plt.title(os.path.basename(tar[i].numpy().decode()))
            plt.imshow(dst.numpy())
            plt.subplot(3, 1, 3)

            res_z = denseNet_model(tf.expand_dims((src - 128) / 255.0, axis=0))
            res_img = load_model.get_decoder_output(res_z)
            res_img = load_model.deNormalize(res_img)
            plt.title('result_image')
            plt.imshow(res_img[0,...])
            num = np.random.random()
            num = str(num)
            plt.savefig('./RenderResult/'+num+'.png')





# 定义optimizer以及checkpoint
lr = tf.Variable(LEARNING_RATE1)
image2sky_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                               beta_1=BETA1, beta_2=BETA2)

checkpoint_dir = "./czy-checkpoint"
if os.path.exists(checkpoint_dir)==False:
    os.makedirs(checkpoint_dir)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt").replace("\\", "/")

checkpoint = tf.train.Checkpoint(image2sky_optimizer=image2sky_optimizer,
                                 denseNet_model=denseNet_model)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# In[18]:

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = './logs/gradient_tape/' + current_time + '/train_input'
validation_log_dir = './logs/gradient_tape/' + current_time + '/validation'
test_log_dir = './logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)



# In[ ]:
test_dataset=[]
# 开始训练
train(train_dataset, test_dataset, 100, checkpoint, checkpoint_prefix)

print("Train Done.")
tf.saved_model.save(denseNet_model,'./Image2SkyModel')

# In[ ]:




