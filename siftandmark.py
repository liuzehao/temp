'''
@Author: haoMax
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293
@Date: 2019-10-24 15:41:50
@LastEditTime: 2019-11-14 10:00:21
@LastEditors: haoMax
@Description: 
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = './249046271.jpg'


surf = cv2.xfeatures2d.SURF_create()

img1 = cv2.imread(imgname1)
kp1, des1 = surf.detectAndCompute(img1,None)#des是描述子

sss=np.ones(np.shape(img1),dtype=np.uint8)
sss[100:350,110:400]=255
print(np.unique(sss))
image=cv2.add(img1,sss)
img3 = cv2.drawKeypoints(image,kp1,image,color=(255,0,255))

cv2.imshow("SURF",image)

cv2.waitKey(0)
cv2.destroyAllWindows()
