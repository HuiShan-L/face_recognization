# -*- coding: utf-8 -*-
"""
将图片数据整理为narray, 
并采用留出法将训练后的数据以8:2的比例划分为训练集和测试集
@author: xs
"""
import os
import cv2
import numpy as np 
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

file_path='./faceImageGray'
filelist=os.listdir(file_path)
data_x=np.empty((5254,64,64),dtype=int)
data_y=np.empty((5254,1),dtype=int)
i=0
#类别标记
Typ=0
j=0
#读取图片信息
for f in filelist:
    
    image_path=file_path+'/'+f
    image_list=os.listdir(image_path)
    for img in image_list:
        j+=1
        im=np.array(Image.open(image_path+'/'+img))
        data_x[i]=im
        data_y[i]=Typ
        i+=1
        print('save'+image_path+'/'+img)
    Typ+=1
    
#留出法样本划分
X_train,X_test,Y_train,Y_test=train_test_split(data_x,data_y,test_size=0.2,random_state=42)
X_train=X_train.astype(np.float32)/255.0
X_test=X_test.astype(np.float32)/255.0
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

#保存数据
np.save('./data/x_tr.npy',X_train)
np.save('./data/y_tr.npy',Y_train)
print('train_data.npy Done!')
np.save('./data/x_te.npy',X_test)
np.save('./data/y_te.npy',Y_test)
print('labels.npy Done!')
