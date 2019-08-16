"""
使用mtcnn算法进行图片人脸检测, 并实现人脸输出

@author: xs
"""
import os
import cv2
from mtcnn.mtcnn import MTCNN

outer_path_file = './aa'
filelist_file = os.listdir(outer_path_file)  # 列举文件夹
detector = MTCNN()
file_put='./faceImageGray'

for f in filelist_file:
    file=outer_path_file+'/'+f  #文件夹名
    put=file_put+'/'+f    #输出文件夹
    filelist=os.listdir(file) #获取
    
    #判断文件是否存在
    if not os.path.exists(put):
        os.makedirs(put) 
     
    j=0
     #获取照片
    for item in filelist:
        src =file+'/'+item
        input_img = cv2.imread(src)
        #需要在mtcnn.py源文件中对detect_faces() 函数进行修改
        detected = detector.detect_faces(input_img)
        if len(detected) > 0:  # 大于0则检测到人脸
            de=detected[0] #可能检测出多张人脸(排除其他噪音)
            x1, y1, w, h = de['box']
            image = input_img[(y1):(y1+h+10), (x1):(x1+w+10)] #
            imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   #灰度处理
            im=cv2.resize(imgray,(64,64))
            cv2.imwrite(put+'/'+'%d.jpg'%(j), im)
            
            print(put+'/'+'%d.jpg'%(j))  
            j+=1

print('Complete all!')