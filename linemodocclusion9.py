#coding=utf-8
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from nets.resnet_v1 import resnetv1
from utils.timer import Timer
import torch
import cv2
import os
import re
import numpy as np
from MeshPly import Meshply
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#create by haoMax
#7.21这个版本采用linemod中的label,然后更换模型为ply,这样有效避免诸多问题和矛盾
#7.24 这个版本专门用来显示8点预测误差了
#9.29 这个版本改变以往的计算方式，跳过单个未标注物体，而不是跳过整张图片
#10.24 专门针对glue做优化
def get_type(class_path,class_name):
    restr='[0-9a-zA-Z]'+'+\.'+class_name
    findtxt = re.compile(restr)
    s=os.listdir(class_path)
    s=" ".join(s)
    s=findtxt.findall(s)[0]
    return s
def read_file(path):
    fid = open(path, 'r')
    f_s =fid.readlines()
    fid.close()
    return f_s
def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 320
    K[1, 1], K[1, 2] = 573.5704, 240
    K[2, 2] = 1.
    return K
def compute_projection(points3d, R, K):
    projections_2d = np.zeros((2, points3d.shape[1]), float)
    camera_projection = (K.dot(R)).dot(points3d)
    projections_2d[0,:] = camera_projection[0,:]/ camera_projection[2,:]
    projections_2d[1,:] = camera_projection[1,:]/ camera_projection[2,:]
    return projections_2d
def show2d(im_name,proj_2d_pr):
    img = cv2.imread(im_name)
    #1.求len长度
    l=len(proj_2d_pr[0])
    #2.改变像素值255
    for i in range(l):
        y=proj_2d_pr[1][i]
        x=proj_2d_pr[0][i]
        #print(x, y)
    #3.改变三通道的像素
        
        x=int(x)
        y=int(y)
        if x>640:
            x=639
        if x<0:
            x=0
        if y>480:
            y=479
        if y<0:
            y=0
        print(x, y)
        img[y,x,0]=255
        img[y,x,1]=255
        img[y,x,2]=255
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()#liu change
def get_all_files(bg_path):
    files = []
    for f in os.listdir(bg_path):
        if os.path.isfile(os.path.join(bg_path, f)):
            files.append(os.path.join(bg_path, f))
        else:
            files.extend(get_all_files(os.path.join(bg_path, f)))
    files.sort(key=lambda x: int(x[-9:-4]))#排序从小到大
    return files
def othertowpont(center,point):
    if point[0]<center[0]:
        x=2*(center[0]-point[0])+point[0]
        if point[1]<center[1]:
            y=point[1]-2*(center[1]-point[1])
        else:
            y=point[1]+2*(center[1]-point[1])
    else:
        x=point[0]-2*(point[0]-center[0])
        if point[1]<center[1]:
            y=point[1]+2*(center[1]-point[1])
        else:
            y=point[1]-2*(point[1]-center[1])
    return (x,y)

def otherpoint(point_c,point_z,point_y):
    
    # cv2.circle(im,point_c,5,(0,0,255),2)
    # cv2.circle(im,point_z,5,(0,0,255),2)
    # cv2.circle(im,point_y,5,(0,0,255),2)
    
    #2.1凭借1点+中心点=1点
    x1,y1=othertowpont(point_c,point_z)
    x2,y2=othertowpont(point_c,point_y)
    # cv2.circle(im,(x1,y1),5,(0,0,255),2)
    # cv2.circle(im,(x2,y2),5,(0,0,255),2)
    # cv2.imshow("img",im)
    # cv2.waitKey(0)
    return ((x1,y1),(x2,y2))
def show_result_3points_glue(cls,bbox,frontp1, frontp2, fcenterp, backp1, backp2, bcenterp,img):
    # img = cv2.imread(filename)
    x1 = frontp1[0]
    y1 = frontp1[1]
    dw1 = frontp1[2]
    dh1 = frontp1[3]

    cx1 = fcenterp[0]
    cy1 = fcenterp[1]

    x3 = frontp2[0]
    y3 = frontp2[1]
    dw2 = frontp2[2]
    dh2 = frontp2[3]

    cv2.circle(img, (x1, y1), 5, (0, 0, 255), 2)
    cv2.circle(img, (cx1, cy1), 5, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 5, (0, 0, 255), 2)

    if cx1 > x1:
        x4 = x1 + dw1
        if cy1 > y1:
            y4 = y1 + dh1
        else:
            y4 = y1 - dh1
    else:
        x4 = x1 - dw1
        if cy1 > y1:
            y4 = y1 + dh1
        else:
            y4 = y1 - dh1
    
    # point3, points2
    if cx1 > x3:
        x2 = x3 + dw2
        if cy1 > y3:
            y2 = y3 + dh2
        else:
            y2 = y3 - dh2
    else:
        x2 = x3 - dw2
        if cy1 > y3:
            y2 = y3 + dh2
        else:
            y2 = y3 - dh2
    # (x4,y4),(x2,y2)=otherpoint((cx1, cy1),(x1, y1),(x3, y3))
    x2=int(x2)
    x4=int(x4)
    y2=int(y2)
    y4=int(y4)
    # cv2.circle(img, (x4, y4), 5, (0, 0, 255), 2)
    # cv2.circle(img, (x2, y2), 5, (0, 0, 255), 2)
    # print("1:",x1,y1)
    # print("2:",x2,y2)
    # print("3:",x3,y3)
    # print("4:",x4,y4)
    # print("c:",cx1,cy1)
    #这里是bbox
    # cv2.putText(img,cls,(int((bbox[0])),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    # cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(0, 255, 0),1)
    x5 = backp1[0]
    y5 = backp1[1]
    dw5 = backp1[2]
    dh5 = backp1[3]

    cx2 = bcenterp[0]
    cy2 = bcenterp[1]

    x7 = backp2[0]
    y7 = backp2[1]
    dw6 = backp2[2]
    dh6 = backp2[3]

    cv2.circle(img, (x5, y5), 5, (0, 255, 255), 2)
    cv2.circle(img, (cx2, cy2), 5, (0, 255, 255), 2)
    cv2.circle(img, (x7, y7), 5, (0, 255, 255), 2)
    # (x8,y8),(x6,y6)=otherpoint((cx2, cy2),(x5, y5),(x7, y7))
    # x8=int(x8)
    # x6=int(x6)
    # y8=int(y8)
    # y6=int(y6)

    # cv2.circle(img, (x8, y8), 5, (0, 255, 255), 2)
    # cv2.circle(img, (x6, y6), 5, (0, 255, 255), 2)
    # # point1, point4
    if cx2 > x5:
        x8 = x5 + dw5
        if cy2 > y5:
            y8 = y5 + dh5
        else:
            y8 = y5 - dh5
    else:
        x8 = x5 - dw5
        if cy2 > y5:
            y8 = y5 + dh5
        else:
            y8 = y5 - dh5
    
    # point3, points2
    if cx2 > x7:
        x6 = x7 + dw6
        if cy2 > y7:
            y6 = y7 + dh6
        else:
            y6 = y7 - dh6
    else:
        x6 = x7 - dw6
        if cy2 > y7:
            y6 = y7 + dh6
        else:
            y6 = y7 - dh6
    # cx = centerp[0]
    # cy = centerp[1]
    
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")


    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    pr_points = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], float)
    # print(pr_points)
    # fang[-1]
    return pr_points
def show_result_3points(cls,bbox,frontp1, frontp2, fcenterp, backp1, backp2, bcenterp,img):
    # img = cv2.imread(filename)
    x1 = frontp1[0]
    y1 = frontp1[1]
    dw1 = frontp1[2]
    dh1 = frontp1[3]

    cx1 = fcenterp[0]
    cy1 = fcenterp[1]

    x3 = frontp2[0]
    y3 = frontp2[1]
    dw2 = frontp2[2]
    dh2 = frontp2[3]

    cv2.circle(img, (x1, y1), 5, (0, 0, 255), 2)
    cv2.circle(img, (cx1, cy1), 5, (0, 0, 255), 2)
    cv2.circle(img, (x3, y3), 5, (0, 0, 255), 2)

    # if cx1 > x1:
    #     x4 = x1 + dw1
    #     if cy1 > y1:
    #         y4 = y1 + dh1
    #     else:
    #         y4 = y1 - dh1
    # else:
    #     x4 = x1 - dw1
    #     if cy1 > y1:
    #         y4 = y1 + dh1
    #     else:
    #         y4 = y1 - dh1
    
    # # point3, points2
    # if cx1 > x3:
    #     x2 = x3 + dw2
    #     if cy1 > y3:
    #         y2 = y3 + dh2
    #     else:
    #         y2 = y3 - dh2
    # else:
    #     x2 = x3 - dw2
    #     if cy1 > y3:
    #         y2 = y3 + dh2
    #     else:
    #         y2 = y3 - dh2
    (x4,y4),(x2,y2)=otherpoint((cx1, cy1),(x1, y1),(x3, y3))
    x2=int(x2)
    x4=int(x4)
    y2=int(y2)
    y4=int(y4)
    # cv2.circle(img, (x4, y4), 5, (0, 0, 255), 2)
    # cv2.circle(img, (x2, y2), 5, (0, 0, 255), 2)
    # print("1:",x1,y1)
    # print("2:",x2,y2)
    # print("3:",x3,y3)
    # print("4:",x4,y4)
    # print("c:",cx1,cy1)
    #这里是bbox
    # cv2.putText(img,cls,(int((bbox[0])),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    # cv2.rectangle(img,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(0, 255, 0),1)
    x5 = backp1[0]
    y5 = backp1[1]
    dw5 = backp1[2]
    dh5 = backp1[3]

    cx2 = bcenterp[0]
    cy2 = bcenterp[1]

    x7 = backp2[0]
    y7 = backp2[1]
    dw6 = backp2[2]
    dh6 = backp2[3]

    cv2.circle(img, (x5, y5), 5, (0, 255, 255), 2)
    cv2.circle(img, (cx2, cy2), 5, (0, 255, 255), 2)
    cv2.circle(img, (x7, y7), 5, (0, 255, 255), 2)
    (x8,y8),(x6,y6)=otherpoint((cx2, cy2),(x5, y5),(x7, y7))
    x8=int(x8)
    x6=int(x6)
    y8=int(y8)
    y6=int(y6)

    # cv2.circle(img, (x8, y8), 5, (0, 255, 255), 2)
    # cv2.circle(img, (x6, y6), 5, (0, 255, 255), 2)
    # # point1, point4
    # if cx2 > x5:
    #     x8 = x5 + dw5
    #     if cy2 > y5:
    #         y8 = y5 + dh5
    #     else:
    #         y8 = y5 - dh5
    # else:
    #     x8 = x5 - dw5
    #     if cy2 > y5:
    #         y8 = y5 + dh5
    #     else:
    #         y8 = y5 - dh5
    
    # # point3, points2
    # if cx2 > x7:
    #     x6 = x7 + dw6
    #     if cy2 > y7:
    #         y6 = y7 + dh6
    #     else:
    #         y6 = y7 - dh6
    # else:
    #     x6 = x7 - dw6
    #     if cy2 > y7:
    #         y6 = y7 + dh6
    #     else:
    #         y6 = y7 - dh6
    # cx = centerp[0]
    # cy = centerp[1]
    
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")


    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    pr_points = np.array([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8], float)
    # print(pr_points)
    # fang[-1]
    return pr_points
def show_result_8points(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
    img = cv2.imread(img)    
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()#liu change
def show_result_8points_test(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
  
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.circle(img, (x1, y1), 1, (0, 0, 255),4)
    cv2.putText(img,'1',(x1, y1),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'2',(x2, y2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'3',(x3, y3),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'4',(x4, y4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'5',(x5, y5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'6',(x6, y6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'7',(x7, y7),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    cv2.putText(img,'8',(x8, y8),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 0, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 0, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 0, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 0, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 0, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    cv2.line(img, (x1, y1), (x5, y5), (255, 0, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 0, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 0, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 0, 0), 1)
def show_result_8points(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
        
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)

    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)

   
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    cv2.putText(img, '1', (x1, y1), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '2', (x2, y2), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '3', (x3, y3), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '4', (x4, y4), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '5', (x5, y5), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '6', (x6, y6), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '7', (x7, y7), font, 1.2, (255, 255, 255), 2) 
    cv2.putText(img, '8', (x8, y8), font, 1.2, (255, 255, 255), 2)
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()#liu change
    return img
def show_result_8points_lunwen(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
        
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)

    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 1)

   
    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (255, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (255, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (255, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (255, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (255, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (255, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (255, 255, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    font = cv2.FONT_HERSHEY_SIMPLEX  
    cv2.line(img, (x1, y1), (x5, y5), (255, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (255, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (255, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (255, 255, 0), 1)
    # cv2.putText(img, '1', (x1, y1), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '2', (x2, y2), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '3', (x3, y3), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '4', (x4, y4), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '5', (x5, y5), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '6', (x6, y6), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '7', (x7, y7), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '8', (x8, y8), font, 1.2, (255, 255, 255), 2)
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()#liu change
    return img
def show_result_8points_nocenter(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,img):
    img = cv2.imread(img)    
    # cv2.line(img, (x1,y1), (x4, y4), 255, 2)
    # cv2.line(img, (x3, y3), (x2, y2), 255, 2)
    x1=int(x1)
    x2=int(x2)
    x3=int(x3)
    x4=int(x4)
    x5=int(x5)
    x6=int(x6)
    x7=int(x7)
    x8=int(x8)
    y1=int(y1)
    y2=int(y2)
    y3=int(y3)
    y4=int(y4)
    y5=int(y5)
    y6=int(y6)
    y7=int(y7)
    y8=int(y8)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # print("1和2")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x1, y1), (x3, y3), (0, 255, 0), 1)
    # print("1和3")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x2, y2), (x4, y4), (0, 255, 0), 1)
    # print("2和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x3, y3), (x4, y4), (0, 255, 0), 1)
    # print("3和4")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.line(img, (x5, y5), (x8, y8), 255, 2)
    # cv2.line(img, (x7, y7), (x6, y6), 255, 2)
    cv2.line(img, (x5, y5), (x6, y6), (0, 255, 0), 1)
    # print("5和6")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x5, y5), (x7, y7), (0, 255, 0), 1)
    # print("5和7")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x6, y6), (x8, y8), (0, 255, 0), 1)
    # print("6和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.line(img, (x7, y7), (x8, y8), (0, 255, 0), 1)
    # print("7和8")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    cv2.line(img, (x1, y1), (x5, y5), (0, 255, 0), 1)
    cv2.line(img, (x2, y2), (x6, y6), (0, 255, 0), 1)
    cv2.line(img, (x3, y3), (x7, y7), (0, 255, 0), 1)
    cv2.line(img, (x4, y4), (x8, y8), (0, 255, 0), 1)
    # cv2.putText(img, '0', (x0, y0), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '1', (x1, y1), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '2', (x2, y2), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '3', (x3, y3), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '4', (x4, y4), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '5', (x5, y5), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '6', (x6, y6), font, 1.2, (255, 255, 255), 2) 
    # cv2.putText(img, '7', (x7, y7), font, 1.2, (255, 255, 255), 2)
    # cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx1, cy1), 2, (0, 0, 255), 1)
    # cv2.circle(img, (cx2, cy2), 2, (255, 0, 255), 1)
    # cv2.imshow('imgoo', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()#liu change
    return img
def demo(net, image_name):
    #print(image_name)
    im = cv2.imread(image_name)
    timer = Timer()
    timer.tic()
    scores, boxes, front_2_1_points, front_2_2_points, front_center, back_2_1_points, back_2_2_points, back_center= im_detect(net, im)
    timer.toc()
    thresh = 0.75  # CONF_THRESH
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    cntr = -1
    
    prs_points=[]#这里存放预测的点
    for cls_ind, cls in enumerate(CLASSES[1:]):
        #cls就是物体的类别
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_front_2_1_points = front_2_1_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_front_2_2_points = front_2_2_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_front_center = front_center[:, 2*cls_ind:2*(cls_ind + 1)]
        cls_back_2_1_points = back_2_1_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_back_2_2_points = back_2_2_points[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_back_center = back_center[:, 2*cls_ind:2*(cls_ind + 1)]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #这里是极大值抑制
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        front_2_1_points_det = cls_front_2_1_points[keep.numpy(), :]
        front_2_2_points_det = cls_front_2_2_points[keep.numpy(), :]
        front_center_det = cls_front_center[keep.numpy(), :]
        back_2_1_points_det = cls_back_2_1_points[keep.numpy(), :]
        back_2_2_points_det = cls_back_2_2_points[keep.numpy(), :]
        back_center_det = cls_back_center[keep.numpy(), :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        inds = [0]
        if len(inds) == 0:
            continue
        else:
            cntr += 1
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            frontp1 = front_2_1_points_det[i, :]
            frontp2 = front_2_2_points_det[i, :]
            fcenterp = front_center_det[i, :]
            brontp1 = back_2_1_points_det[i, :]
            brontp2 = back_2_2_points_det[i, :]
            bcenterp = back_center_det[i, :]
            img = cv2.imread(image_name)
            # if cls_ind==7:
            #     pr_points = show_result_3points_glue(cls,bbox,frontp1, frontp2, fcenterp, brontp1, brontp2, bcenterp,img)
            # else:
            #     pr_points = show_result_3points(cls,bbox,frontp1, frontp2, fcenterp, brontp1, brontp2, bcenterp,img)
            pr_points = show_result_3points_glue(cls,bbox,frontp1, frontp2, fcenterp, brontp1, brontp2, bcenterp,img)
            prs_points.append(pr_points)
            
            #show_result_3points(brontp1, brontp2, bcenterp, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()#liu change
    return prs_points
def get_3D_corners(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],#1
                        [min_x, min_y, max_z],#2
                        [min_x, max_y, min_z],#3
                        [min_x, max_y, max_z],#4
                        [max_x, min_y, min_z],#5
                        [max_x, min_y, max_z],#6
                        [max_x, max_y, min_z],#7
                        [max_x, max_y, max_z]])#8
    return corners
def get_3D_corners_test(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, max_y, min_z],#3
                        [min_x, max_y, max_z],#4
                        [max_x, min_y, min_z],#5
                        [max_x, min_y, max_z]])#4
    return corners
def get_3D_corners3(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],#1
                        [max_x, min_y, min_z],#5
                        [min_x, min_y, max_z],#2
                        [max_x, min_y, max_z],#6
                        [min_x, max_y, min_z],#3
                        [max_x, max_y, min_z],#7
                        [min_x, max_y, max_z],#4
                        [max_x, max_y, max_z]])#8
    return corners
def get_3D_corners3_test(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, max_z],#2
                        [max_x, min_y, max_z],#6
                        [min_x, max_y, min_z],#3
                        [max_x, max_y, min_z]])#7
    return corners
def get_3D_corners2(vertices):
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],#1
                        [min_x, max_y, min_z],#3
                        [max_x, min_y, min_z],#5
                        [max_x, max_y, min_z],#7
                        [min_x, min_y, max_z],#2
                        [min_x, max_y, max_z],#4
                        [max_x, min_y, max_z],#6    
                        [max_x, max_y, max_z]])#8
    return corners
def test_RT(t_gt):
    Tx=t_gt[0]
    Ty=t_gt[1]
    Tz=t_gt[2]
    K=get_camera_intrinsic()
    Cx=K[0][0]*(Tx/Tz)+320
    Cy=K[1][1]*(Ty/Tz)+240
    return (Cx,Cy)
def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)
###路径###
xyz_path_base='../data/OcclusionData/models/'#换成xyz模型
CLASSES = ('__background__',
           'ape','can','cat','driller','duck','eggbox','glue','holepuncher')#liu change
color_list=[(0,255,0),(0,0,255),(255,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0),(255,255,255)]
pose_path='../data/OcclusionData/poses/'
label_linemod_path_base='../data/linemodocculution/'
im_path="../data/OcclusionData/RGB-D/rgb_noseg_all"#遮挡数据集路径
ply_path_base='../data/ply/'#模型基本路径
# model_path='../output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_320000.pth'
model_path='../output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_300000.pth'
if __name__ == '__main__':
    ims_path=get_all_files(im_path)#读入所有测试图片
    errs_2d_ape = []
    errs_3d_ape = []
    errs_2d_can = []
    errs_3d_can = []
    errs_2d_cat = []
    errs_3d_cat = []
    errs_2d_driller = []
    errs_3d_driller = []
    errs_2d_duck = []
    errs_3d_duck = []
    errs_2d_eggbox = []
    errs_3d_eggbox = []
    errs_2d_glue = []
    errs_3d_glue = []
    errs_2d_holepuncher = []
    errs_3d_holepuncher = []
    errs_2d=[errs_2d_ape,errs_2d_can,errs_2d_cat,errs_2d_driller,errs_2d_duck,errs_2d_eggbox,errs_2d_glue,errs_2d_holepuncher]
    errs_3d=[errs_3d_ape,errs_3d_can,errs_3d_cat,errs_3d_driller,errs_3d_duck,errs_3d_eggbox,errs_3d_glue,errs_3d_holepuncher]
    meshlist=[]
    #先把模型读到内存中
    for z in range(1,len(CLASSES)):
        ply_path=ply_path_base+CLASSES[z]+'.ply'
        mesh = Meshply(ply_path)
        meshlist.append(mesh)
    net = resnetv1(num_layers=101)
    saved_model = model_path
    net.create_architecture(9, tag='default', anchor_scales=[8, 12, 16])  # class 7 #fang liu 9
    net.load_state_dict(torch.load(saved_model))
    net.eval()
    net.cuda()
    for im_name in ims_path:
        print(im_name)
        #2019.7.7 构建异常跳过没有标签的图片
        flag_preerro=0
        erro_num=0
        img_name=im_name[-9:-4]
        # print(net)
        prs_points = demo(net, im_name)#这里调用了net 和im_name进行预测,关键是解析出pr.point然后跟xml去比较
        img = cv2.imread(im_name)
        for flag_u in range(1,len(CLASSES)):
            #print(CLASSES[flag_u])
            #1.读入模型 这里改为ply模型
            
            mesh = meshlist[flag_u-1]
            vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
            #print("3D-point:",vertices) #通过验证一,格式正确
             
             #2.读入测试points
             #格式化名字
            img_name_label = '%06d' % int(img_name)
            label_linemod_path=label_linemod_path_base+CLASSES[flag_u]+'/'+img_name_label+'.txt'
            f=open(label_linemod_path,'r')
            line=f.readline()
            line=line.split()
            label_linemod_points=line[3:19]
            label_linemod_points=np.array(map(float,label_linemod_points))
            label_linemod_points[::2]=label_linemod_points[::2]*640
            label_linemod_points[1::2]=label_linemod_points[1::2]*480

            for zz in range(len(label_linemod_points[::2])):
                if label_linemod_points[::2][zz]>640 or label_linemod_points[1::2][zz]>480:
                    erro_num=erro_num+1
                    print("problem picture is:",im_name)
                    print("problem num is：",CLASSES[flag_u])
                    flag_preerro=1
                    continue
            if flag_preerro==1:
                continue#跳过整张图片的代码---->跳过单个物体
            
            gt_points=label_linemod_points.reshape(8,2)
            #print("gt_points:",gt_points)
            #这个位置交换一下gt_points变成正常顺序
            #gt_points[0],gt_points[1],gt_points[2],gt_points[3],gt_points[4],gt_points[5],gt_points[6],gt_points[7]=gt_points[0],gt_points[2],gt_points[4],gt_points[6],gt_points[1],gt_points[3],gt_points[5],gt_points[7]
            # print("gt_points_exchange:",gt_points)
            # print('gt_pointsx:',label_linemod_points[::2])
            # print('gt_pointsy:',label_linemod_points[1::2])
            #显示bbox3d
            # img=show_result_8points_nocenter(gt_points[0][0],gt_points[0][1],gt_points[1][0],gt_points[1][1],gt_points[2][0],gt_points[2][1],gt_points[3][0],gt_points[3][1],gt_points[4][0],gt_points[4][1],gt_points[5][0],gt_points[5][1],gt_points[6][0],gt_points[6][1],gt_points[7][0],gt_points[7][1],im_name)

            #通过测试,8point没问题
    #3.处理预测点prs_points
            K=get_camera_intrinsic()
            pr_points = prs_points[flag_u-1].reshape(8,2)
            #print("pr_points:",pr_points)
            # print('pr_pointsx:',prs_points[flag_u-1][::2])
            # print('pr_pointsy:',prs_points[flag_u-1][1::2])
            # img=show_result_8points(pr_points[0][0],pr_points[0][1],pr_points[1][0],pr_points[1][1],pr_points[2][0],pr_points[2][1],pr_points[3][0],pr_points[3][1],pr_points[4][0],pr_points[4][1],pr_points[5][0],pr_points[5][1],pr_points[6][0],pr_points[6][1],pr_points[7][0],pr_points[7][1],img)
            #论文显示
            # img=show_result_8points_lunwen(pr_points[0][0],pr_points[0][1],pr_points[1][0],pr_points[1][1],pr_points[2][0],pr_points[2][1],pr_points[3][0],pr_points[3][1],pr_points[4][0],pr_points[4][1],pr_points[5][0],pr_points[5][1],pr_points[6][0],pr_points[6][1],pr_points[7][0],pr_points[7][1],img)
            #显示bbox3d_pr
            # show_result_8points_test(pr_points[0][0],pr_points[0][1],pr_points[1][0],pr_points[1][1],pr_points[2][0],pr_points[2][1],pr_points[3][0],pr_points[3][1],pr_points[4][0],pr_points[4][1],pr_points[5][0],pr_points[5][1],pr_points[6][0],pr_points[6][1],pr_points[7][0],pr_points[7][1],img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()#liu change
            corners3D_pr = get_3D_corners3(vertices)#这里需要根据训练集的选择有所改变，如果是occulution2015那个数据集训练的要改成get_3D_corners不带3版本
            corners3D_gt = get_3D_corners3(vertices)#必须旋转模型才行,两种结果模型不一样
    #4.预测R|T
    #测试挑4个点
            # corners3D_pr_test = get_3D_corners_test(vertices)
            # corners3D_gt_test = get_3D_corners3_test(vertices)#必须旋转模型才行,两种结果模型不一样
            # pr_points_test= np.zeros((4, 2), dtype='float64')
            # pr_points_test[0]=pr_points[3]
            # pr_points_test[1]=pr_points[4]
            # pr_points_test[2]=pr_points[5]
            # pr_points_test[3]=pr_points[5]
            _, R_pre, t_pre = cv2.solvePnP(corners3D_pr, pr_points, K, None)
            R_mat_pre, _ = cv2.Rodrigues(R_pre)
            #print("t_pre[2]1:",t_pre[2])
            # gt_points_test= np.zeros((4, 2), dtype='float64')
            # gt_points_test[0]=gt_points[3]
            # gt_points_test[1]=gt_points[4]
            # gt_points_test[2]=gt_points[5]
            # gt_points_test[3]=gt_points[6]

            _, R_gt, t_gt = cv2.solvePnP(corners3D_gt, gt_points, K, None)
            R_mat_gt, _ = cv2.Rodrigues(R_gt)
            Cx,Cy=test_RT(t_gt)

            # print("t_gt:",t_gt)
            # print("gt_points:",gt_points)
            # show_result_8points(Cx,Cy,gt_points[0][0],gt_points[0][1],gt_points[1][0],gt_points[1][1],gt_points[2][0],gt_points[2][1],gt_points[3][0],gt_points[3][1],gt_points[4][0],gt_points[4][1],gt_points[5][0],gt_points[5][1],gt_points[6][0],gt_points[6][1],gt_points[7][0],gt_points[7][1],im_name)
            #show_result_8points(Cx,Cy,pr_points[0][0],pr_points[0][1],pr_points[1][0],pr_points[1][1],pr_points[2][0],pr_points[2][1],pr_points[3][0],pr_points[3][1],pr_points[4][0],pr_points[4][1],pr_points[5][0],pr_points[5][1],pr_points[6][0],pr_points[6][1],pr_points[7][0],pr_points[7][1],im_name)
            # cv2.imshow('img', im_name)
            # cv2.waitKey(0)
            #t_pre[2]=t_gt[2]
            #print("t_pre[2]2:",t_pre[2])

            Rt_gt = np.concatenate((R_mat_gt, t_gt), axis=1)
            Rt_pr = np.concatenate((R_mat_pre, t_pre), axis=1)
            #1.算出中心点的坐标
            corners3D = get_3D_corners(vertices)
            center_point=(corners3D[7]+corners3D[0])/2
            # print(corners3D[7])
            # print(corners3D[0])
            # print(center_point)
            center_point=np.insert(center_point,3,[1],axis=0).reshape(-1,1)
            #print(center_point)
            #2.计算中心点投影
            proj_center = compute_projection(center_point,Rt_gt,K)
            #3.比较
            Cx,Cy=test_RT(t_gt)
            # print("proj_center:",proj_center)
            # print("Cx,Cy:",(Cx,Cy))
            # print("R_gt:",R_gt)
            # print("R_pre:",R_pre)
            # print("t_pre:",t_pre)
            # print("t_gt:",t_gt)
            #T做差寻找规律
            # print(CLASSES[flag_u])
            #print("human erro:",t_gt-t_pre)
    #5.计算2d投影
    #2019.7.7 构建异常可能是预测错误,重新计算这张图片(先跳过)
            try:
                proj_2d_gt = compute_projection(vertices,Rt_gt,K)
                proj_2d_pr = compute_projection(vertices,Rt_pr, K)
            except IndexError as e:
                print("pre erro:",img_name)
                flag_preerro=1
                break
#6.显示2d投影
            #print("gt",proj_2d_gt)
            #print("pr",proj_2d_pr)
            #show2d(im_name,proj_2d_gt)#大致通过验证,等待转换图像获取后再验证
            #show2d(im_name,proj_2d_pr)
            #心得:使用label外参数需要640-x,使用自己的外参数不需要,因为是直接估计的
#7.计算3d
#8.计算2d和3d误差
            #8.1计算2d误差
            norm = np.linalg.norm(proj_2d_gt - proj_2d_pr, axis=0)
            #print("norm:",norm)
            pixel_dist = np.mean(norm)
            # print("pixel2D_dist",pixel_dist)#显示2d结果
            # if(pixel_dist > 5.0):
            #     print('!!!!!', im_name)
            errs_2d[flag_u-1].append(pixel_dist)

            # #8.2计算3d误差
            transform_3d_gt = compute_transformation(vertices, Rt_gt)
            transform_3d_gt=np.array(transform_3d_gt)
            #transform_3d_gt[0]=-transform_3d_gt[0]
            transform_3d_pr = compute_transformation(vertices, Rt_pr)
            transform_3d_pr=np.array(transform_3d_pr)

            #8.3绘制3d图像,查看问题在哪里
            # savedStdout = sys.stdout  #保存标准输出流
            # with open('./3dpoints_gt_z.txt', 'wt') as file:
            #     sys.stdout = file  #标准输出重定向至文件
            #     np.set_printoptions(threshold='nan')#numpy全打印
            #     print(transform_3d_gt[2].tolist())
            # sys.stdout = savedStdout  #恢复标准输出流
            # exit()
            #显示
            #GT数据
            # gt_x=transform_3d_gt[0].tolist()
            # gt_y=transform_3d_gt[1].tolist()
            # gt_z=transform_3d_gt[2].tolist()
            # #pr数据
            # pr_x=transform_3d_pr[0].tolist()
            # pr_y=transform_3d_pr[1].tolist()
            # pr_z=transform_3d_pr[2].tolist()
            # #开始绘图
            # fig=plt.figure(dpi=120)
            # ax=fig.add_subplot(111,projection='3d')
            # #标题
            # plt.title('point cloud')
            # #利用xyz的值，生成每个点的相应坐标（x,y,z）
            # ax.scatter(gt_x,gt_y,gt_z,c='b',marker='.',s=1,linewidth=0,alpha=0.5,cmap='spectral')
            # ax.scatter(pr_x,pr_y,pr_z,c='r',marker='.',s=1,linewidth=0,alpha=0.5,cmap='spectral')
            # ax.axis('scaled')          
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # plt.show()
            norm3d          = np.linalg.norm(transform_3d_gt - transform_3d_pr, axis=0)
            vertex_dist     = np.mean(norm3d)
            # print("vertex3D_dist",vertex_dist)#显示3d结果
            errs_3d[flag_u-1].append(vertex_dist)
        #2019.7.7 构建异常跳过没有标签的图片
        if flag_preerro==1:
            continue
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    # print("errs_3d",errs_3d)
    # print("errs_2d",errs_2d)
#9.加入循环,修改为计算平均值
    #print(erro_num)
    px_threshold = 5.
    eps = 1e-5
    #errs_3d_ape,errs_3d_can,errs_3d_cat,errs_3d_driller,errs_3d_duck,errs_3d_eggbox,errs_3d_glue,errs_3d_holepuncher
    diam = [0.103,0.202,0.155,0.262,0.109,0.176364,0.176,0.162]
    for u in range(len(errs_3d)):
        print("num of every class:",len(errs_2d[u]))
        error_count = len(np.where(np.array(errs_2d[u]) > px_threshold)[0])
        acc = len(np.where(np.array(errs_2d[u-1]) <= px_threshold)[0]) * 100.0 / (len(errs_2d[u])+eps)
        acc3d10     = len(np.where(np.array(errs_3d[u]) <= diam[u] * 0.1)[0]) * 100. / (len(errs_3d[u])+eps)
        print('Test finish! Object is:',CLASSES[u+1])
        print('Acc using {} px 2D projection = {:.2f}%'.format(px_threshold, acc))
        print('Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam[u] * 0.1, acc3d10))
