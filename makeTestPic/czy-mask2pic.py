#-*-coding:utf-8-*-
import cv2
import numpy as np
import os
import glob
import random
import MakeMask
import exr2png
skybox_dir = '/home/czy/MyWorkSpace/Deep-Light/MyNet/40000data/skybox/'
mask_dir = '/home/czy/MyWorkSpace/Deep-Light/MyNet/40000data/mask/'
skybox_files_path = glob.glob(skybox_dir + '*.exr')
skybox_files_path.sort()
mask_files_path = glob.glob(mask_dir+'*.png')
mask_files_path.sort()

dst_dir = './NewDataSet'
if os.path.exists(dst_dir)==False:
    os.makedirs(dst_dir)

index = 0
for i in range(len(mask_files_path)):
    skybox  = skybox_files_path[i] # get one skyBox exr
    mask_set = random.sample(mask_files_path,5) # random choose 5 mask to synthesize the pic
    gray_list = list(np.random.uniform(0.08,0.18,3))

    for mask in mask_set:
        for gray in gray_list:
            index +=1
            synPic = MakeMask.genSynthetic(skybox,mask,gray)
            skybox_png = exr2png.exr2png(gray,skybox).astype(int)
            synPic_save_dir = dst_dir + '/synPic/' + str(index) + '.png'
            skyBox_save_dir = dst_dir + '/skybox/' + str(index) + '.png'
            cv2.imwrite(synPic_save_dir,synPic)
            cv2.imwrite(skyBox_save_dir,skybox_png)






