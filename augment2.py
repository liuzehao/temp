'''
@Author: haoMax
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293
@Date: 2019-11-05 20:28:53
@LastEditTime: 2019-11-05 21:15:30
@LastEditors: haoMax
@Description: 
'''
import tensorflow as tf
import os
import random
# import cv2
source_file="./tmp/train/"       #原始文件地址
source_file_mask="./tmp/mask/"
target_file="./tmp/out/jpg/"  #修改后的文件地址
target_file_mask="./tmp/out/mask/"  #修改后的文件地址
num=50                  #产生图片次数
 
if not os.path.exists(target_file):  #如果不存在target_file，则创造一个
    os.makedirs(target_file)
 
file_list=os.listdir(source_file)   #读取原始文件的路径
file_list_mask=os.listdir(source_file_mask)
with tf.Session() as sess:
    for i in range(num):
 
        max_random=len(file_list)-1
        a = random.randint(0, max_random)          #随机数字区间
        image_raw_data=tf.gfile.FastGFile(source_file+file_list[a],"rb").read()#读取图片
        image_raw_data_mask=tf.gfile.FastGFile(source_file_mask+file_list_mask[a],"rb").read()#读取图片
        print("正在处理：",file_list[a])
        
        image_data=tf.image.decode_jpeg(image_raw_data)
        image_data_mask=tf.image.decode_png(image_raw_data_mask)
        if i%2==1:
            filpped=tf.image.random_flip_left_right(image_data)   #随机左右翻转
            filpped_mask=tf.image.random_flip_left_right(image_data_mask)
        else:
            filpped=tf.image.random_flip_up_down(image_data)    #随机上下翻转
            filpped_mask=tf.image.random_flip_left_right(image_data_mask)
 
        adjust=tf.image.random_brightness(filpped,0.4)      #随机调整亮度
 
        image_data=tf.image.convert_image_dtype(adjust,dtype=tf.uint8)
        image_data_mask=tf.image.convert_image_dtype(filpped_mask,dtype=tf.uint8)
        encode_data=tf.image.encode_jpeg(image_data)
        encode_data_mask=tf.image.encode_png(image_data_mask)

        with tf.gfile.GFile(target_file+str(i)+".jpg","wb") as f:
            f.write(encode_data.eval())
        with tf.gfile.GFile(target_file_mask+str(i)+".png","wb") as f:
            f.write(encode_data_mask.eval())
print("图像增强完毕")
 
 
 