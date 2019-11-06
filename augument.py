'''
@Author: haoMax
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293
@Date: 2019-10-10 11:15:18
@LastEditTime: 2019-10-18 15:20:53
@LastEditors: haoMax
@Description: 
'''
# -*- coding: utf-8 -*-
 
import cv2
from math import *
import numpy as np
import os
# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))
test_path='./logs'
rotate=[90,180,270]
rotate_name=["_a","_b","_c"]
for i in os.listdir(test_path):
    path=os.path.join(test_path,i)

    (filename,extension) = os.path.splitext(i)
    # print(filename)
    img=cv2.imread(path)
    for t in range(len(rotate)): 
        imgRotation = rotate_bound_white_bg(img, rotate[t])
        rename=filename+rotate_name[t]+'.png'
        rename_path=os.path.join(test_path,rename)
        print(rename_path)
        cv2.imwrite(rename_path,imgRotation)
# img = cv2.imread("./logs/bill_214.png")
# imgRotation = rotate_bound_white_bg(img, 90)
# cv2.imwrite("./logs/bill_212_rotatec.jpg",imgRotation) 
# cv2.imshow("img",img)
# cv2.imshow("imgRotation",imgRotation)
# cv2.waitKey(0)
