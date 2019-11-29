#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import os

#
# # 指定训练使用的GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

Mydevice = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(Mydevice[0], True)
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import load_model
import random
import densenet
from PIL import Image

# In[2]:


BUFFER_SIZE = 100
BATCH_SIZE = 1
INPUT_IMG_WIDTH = 224
INPUT_IMG_HEIGHT = 224
TARGET_IMG_WIDTH = 128
TARGET_IMG_HEIGHT = 32
PATH_TRAIN_INPUT_IMAGE = "../../dataSample-for-training/image2sky/input_train"
PATH_TRAIN_TARGET_IMAGE = "../../dataSample-for-training/image2sky/target_train"
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


# In[4]:


# 计算立体角矩阵，暂不启用
# @tf.function
# def solid_angles(target_image):
#     height = target_image.shape[1]
#     width = target_image.shape[2]

#     d_omega = np.zeros((height, width, 3))

#     for i in range(height):
#         for j in range(width):
#             angleN = ((height - i) / height) * (math.pi / 2)
#             angleS = ((height - i - 1) / height) * (math.pi / 2)
#             angleE = ((j + 1) / width) * (2 * math.pi)
#             angleW = (j / width) * (2 * math.pi)
#             d_omega[i][j][0] = (math.sin(angleN) - math.sin(angleS)) * (angleE - angleW)
#             d_omega[i][j][1] = (math.sin(angleN) - math.sin(angleS)) * (angleE - angleW)
#             d_omega[i][j][2] = (math.sin(angleN) - math.sin(angleS)) * (angleE - angleW)

#     d_omega = tf.convert_to_tensor(d_omega, tf.float32)

#     return d_omega


# In[5]:


# 计算loss
def get_image2sky_loss(target_z, output_z):
    lossZ = tf.square(target_z - output_z)
    lossZ = tf.reduce_sum(lossZ)
    lossZ = tf.sqrt(lossZ)
    
    lossC = load_model.get_decoder_output(target_z) - load_model.get_decoder_output(output_z)
    lossC = tf.reduce_sum(tf.abs(lossC))
    
    return lossZ, lossC, lossZ + LAMBDA_C * lossC


# In[6]:


#定义loss矩阵，用来累计一个epoch内每个batch的loss和
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

loss_z = tf.keras.metrics.Mean('loss_z', dtype=tf.float32)
loss_c = tf.keras.metrics.Mean('loss_c', dtype=tf.float32)


# In[ ]:


# 随机从训练集和测试集中各抽取5组数据，显示每5个epoch后的训练效果
img_show  = []
target_img_show = []
target_z_show = []
recon_img = []

train_exrs = os.listdir(PATH_TRAIN_TARGET_IMAGE)
train_exrs.sort()
train_img_dirs = os.listdir(PATH_TRAIN_INPUT_IMAGE)
train_img_dirs.sort()

fileNum = len(train_exrs)
index = np.array(range(fileNum))
np.random.shuffle(index)
index = index[:5]


#这个函数 对每一个 128 * 32的Skybox 取出其对应的 224 * 224 的crop的图片
for i in index:
    exr_img_path = os.path.join(PATH_TRAIN_TARGET_IMAGE, train_exrs[i]).replace("\\", "/")
    target_img_show.append(load_model.load(exr_img_path).numpy())
    target_z_show.append(load_model.get_encoder_output(exr_img_path))
    recon_img.append(load_model.deNormalize(load_model.get_decoder_output(target_z_show[-1])[0]))
    
    crop_img_dir_path = os.path.join(PATH_TRAIN_INPUT_IMAGE, train_img_dirs[i]).replace("\\", "/")
    crop_imgs = os.listdir(crop_img_dir_path)
    crop_imgs.sort()
    j = random.randint(0, 9)
    crop_img_path = os.path.join(crop_img_dir_path, crop_imgs[j]).replace("\\", "/")
    img_show.append(load_image_train(crop_img_path))
    
    print(exr_img_path)
    print(crop_img_path)

    
test_img_show = []
test_target_img_show = []
test_target_z_show = []
test_recon_img = []

test_exrs = os.listdir(PATH_TEST_TARGET_IMAGE)
test_exrs.sort()
test_img_dirs = os.listdir(PATH_TEST_INPUT_IMAGE)
test_img_dirs.sort()

fileNum = len(test_exrs)
index = np.array(range(fileNum))
np.random.shuffle(index)
index = index[:5]

for i in index:
    test_exr_img_path = os.path.join(PATH_TEST_TARGET_IMAGE, test_exrs[i]).replace("\\", "/")
    test_target_img_show.append(load_model.load(test_exr_img_path).numpy())
    test_target_z_show.append(load_model.get_encoder_output(test_exr_img_path))
    test_recon_img.append(load_model.deNormalize(load_model.get_decoder_output(test_target_z_show[-1])[0]))
    
    test_crop_img_dir_path = os.path.join(PATH_TEST_INPUT_IMAGE, test_img_dirs[i]).replace("\\", "/")
    test_crop_imgs = os.listdir(test_crop_img_dir_path)
    test_crop_imgs.sort()
    
    j = random.randint(0, 9)
    test_crop_img_path = os.path.join(test_crop_img_dir_path, test_crop_imgs[j]).replace("\\", "/")
    test_img_show.append(load_image_train(test_crop_img_path))
    
    print(test_exr_img_path)
    print(test_crop_img_path)
    
def show():
    print('Train Dataset Result:')
    print(' ')
    for i in range(5):
        plt.figure(figsize=(20, 14))
        plt.subplot(4, 1, 1)
        plt.title('iput_image')
        plt.imshow(img_show[i].numpy().astype(int))
        plt.subplot(4, 1, 2)
        plt.title('target_image')
        plt.imshow(target_img_show[i])
        plt.subplot(4, 1, 3)
        plt.title('reconstruct_image')
        plt.imshow(recon_img[i].numpy())
        plt.subplot(4, 1, 4)
        res_z = denseNet_model(tf.expand_dims((img_show[i] - 128) / 255.0, axis=0))
        res_img = load_model.get_decoder_output(res_z)
        res_img = load_model.deNormalize(res_img[0])
        plt.title('result_image')
        plt.imshow(res_img.numpy())

        plt.show()

        print("target_z")
        print(target_z_show[i].numpy())
        print("result_z")
        print(res_z.numpy())
        
        
    print('Test Dataset Result:')
    print(' ')
    for i in range(5):
        plt.figure(figsize=(20,14))
        plt.subplot(4, 1, 1)
        plt.title('input_image')
        plt.imshow(test_img_show[i].numpy().astype(int))
        plt.subplot(4, 1, 2)
        plt.title('target_image')
        plt.imshow(test_target_img_show[i])
        plt.subplot(4, 1, 3)
        plt.title('reconstruct_image')
        plt.imshow(test_recon_img[i].numpy())
        plt.subplot(4, 1, 4)
        test_res_z = denseNet_model(tf.expand_dims((test_img_show[i] - 128) / 255.0, axis=0))
        test_res_img = load_model.get_decoder_output(test_res_z)
        test_res_img = load_model.deNormalize(test_res_img[0])
        plt.title('result_image')
        plt.imshow(test_res_img.numpy())
        
        plt.show()
        
        print('target_z')
        print(test_target_z_show[i].numpy())
        print('result_z')
        print(test_res_z.numpy())
        
        print(" ")
        print(" ")
        


# In[8]:


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
    for epoch in range(0, epochs):
        start = time.time()
        for input_image_path, target_z in train_dataset:
            input_image.clear()
            for i in range(BATCH_SIZE):
                input_image.append((load_image_train(input_image_path[i].numpy().decode()) -128) / 255.0)
            input_image_tf = tf.stack(input_image)
            
            train_step(input_image_tf, target_z)
        
        for input_image_path, target_z in test_dataset:
            input_image.clear()   
            for i in range(BATCH_SIZE):
                input_image.append(load_image_train(input_image_path[i].numpy().decode()) - 128 / 255)
            input_image_tf = tf.stack(input_image)
            
            test_step(input_image_tf, target_z)
            
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('loss_z', loss_z.result(), step=epoch)
            tf.summary.scalar('loss_c', loss_c.result(), step=epoch)
            
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
        
        train_loss.reset_states()
        test_loss.reset_states()
        loss_z.reset_states()
        loss_c.reset_states()
        
        # if epoch % 5 == 0:
        #     show()
        
        checkpoint.save(file_prefix = checkpoint_prefix)


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


#加载训练集数据
for i in range(len(train_input_image_dirs)):
    train_input_images_path = os.path.join(PATH_TRAIN_INPUT_IMAGE, train_input_image_dirs[i])
    train_input_images = os.listdir(train_input_images_path)
    train_input_images.sort()

    for train_image in train_input_images:
        train_image_path = os.path.join(train_input_images_path, train_image).replace("\\", "/")
        train_dataset_input_image.append(tf.constant(train_image_path))
    for j in range(len(train_input_images)):
        train_target_image_path = os.path.join(PATH_TRAIN_TARGET_IMAGE, train_target_images[i]).replace("\\", "/")
        train_dataset_target_z.append(tf.squeeze(load_model.get_encoder_output(train_target_image_path)))


# In[13]:


#加载测试集数据
for i in range(len(test_input_image_dirs)):
    test_input_images_path = os.path.join(PATH_TEST_INPUT_IMAGE, test_input_image_dirs[i])
    test_input_images = os.listdir(test_input_images_path)
    test_input_images.sort()
    
    for test_image in test_input_images:
        test_image_path = os.path.join(test_input_images_path, test_image).replace("\\", "/")
        test_dataset_input_image.append(tf.constant(test_image_path))
    for j in range(len(test_input_images)):
        test_target_image_path = os.path.join(PATH_TEST_TARGET_IMAGE, test_target_images[i]).replace("\\", "/")
        test_dataset_target_z.append(tf.squeeze(load_model.get_encoder_output(test_target_image_path)))


# In[14]:


#训练集数据input_image, target_z打包
train_dataset_input_image = tf.data.Dataset.from_tensor_slices(train_dataset_input_image)
train_dataset_target_z = tf.data.Dataset.from_tensor_slices(train_dataset_target_z)

train_dataset = tf.data.Dataset.zip((train_dataset_input_image, train_dataset_target_z))

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


# In[15]:


#测试集数据input_image, target_z打包
test_dataset_input_image = tf.data.Dataset.from_tensor_slices(test_dataset_input_image)
test_dataset_target_z = tf.data.Dataset.from_tensor_slices(test_dataset_target_z)

test_dataset = tf.data.Dataset.zip((test_dataset_input_image, test_dataset_target_z))

test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# In[16]:


# 定义模型对象
denseNet_model = densenet.DenseNet(10, 48, 4, 64, [6, 12, 36, 24], "channels_last",
                                weight_decay=1e-7, pool_initial=True)


# In[17]:


# 定义optimizer以及checkpoint
lr = tf.Variable(LEARNING_RATE1)
image2sky_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, 
                                                beta_1=BETA1, beta_2=BETA2)
checkpoint_dir = "./checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt").replace("\\", "/")

checkpoint = tf.train.Checkpoint(image2sky_optimizer=image2sky_optimizer,
                                    denseNet_model=denseNet_model)

checkpoint.restore(tf.train.latest_checkpoint('./tf-densenet161'))


# In[18]:


# 用于tensorboard显示loss曲线
log_dir = "./logs"
train_summary_writer = tf.summary.create_file_writer(log_dir)


# In[ ]:

# 开始训练
train(train_dataset, test_dataset, 600, checkpoint, checkpoint_prefix)


print("Train Done.")



# In[ ]:




