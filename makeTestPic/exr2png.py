import cv2
import numpy as np

def exr2png(gray,img):

    # print(img)
    # cv2.imshow('1', img)
    img = img.numpy()
    img[img < 0] = 0
    delt = 0.001
    # print(img.shape)
    sum = img.shape[0] * img.shape[1]
    mid = np.log(delt + img)
    # print(mid)
    logValue = mid.sum() / (3 * sum)
    final_v = np.exp(logValue)
    # print(final_v)


    Lxy = gray / final_v * img
    # print(Lxy)
    L = Lxy / (1. + Lxy)
    # print(L)
    # cv2.imshow('2', L)
    L = L * 255
    # L= L[0:32,:,:]
    # print(L.shape)

    return L

# exr2png(r"C:\Users\lenovo\Desktop\ui\UI_image2sky\result\skyResult.exr",r".\skybox.png")
