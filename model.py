# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:06:33 2019

@author: xs
"""

import cv2
import numpy as np
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split


#%% 导入数据

x_tr=np.load('./data/x_tr.npy')
y_tr=np.load('./data/y_tr.npy')

#划分训练集和验证集
x_tr,x_va,y_tr,y_va=train_test_split(x_tr,y_tr,test_size=0.1,random_state=42)
train = tf.train.slice_input_producer([x_tr,y_tr], shuffle=False)
print("Data Compelete!")

#%%

#每个批次的大小
batch_size = 100
#输入x格式64*64
size=64 

#生成batch数据
def get_batch(train,batch_size):
    batch_xs,batch_ys =  tf.train.shuffle_batch(train,batch_size=batch_size,num_threads=1, capacity=64, min_after_dequeue=1)
    with tf.Session() as sess:
        # 线程的协调器
        coord = tf.train.Coordinator()
        # 开始在图表中收集队列运行器
        threads = tf.train.start_queue_runners(sess, coord)
        batch_xs_v, batch_ys_v = sess.run([batch_xs, batch_ys])
        # 请求线程结束
        coord.request_stop()
        # 等待线程终止
        coord.join(threads)
    return batch_xs_v,batch_ys_v
 
#参数概要(tensorboard 输出)
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图
        
#%% CNN定义
        
#初始化权值
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #x input tensor of shape `[batch, in_height, in_width, in_channels]`
    #W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1] 窗口大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义输入变量x,y
with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,64,64],name='x-input')
    y = tf.placeholder(tf.float32,[None,10],name='y-input')
    with tf.name_scope('x_image'):
        #[batch, in_height, in_width, in_channels]`
        x_image = tf.reshape(x,[-1,64,64,1],name='x_image')

#%% CNN搭建
'''
CNN 概览:
    输入:64*64
    Conv1:64*64*32 Pooling1:32*32*32
    Conv2:32*32*32 Pooling2:16*16*64
    Conv3:16*16*64 Pooling3:8*8*64
    Full1:8*8*64,512
    Full2:512,10  
'''
        
#Conv1&Pool1
keep_prob_1 = tf.placeholder(tf.float32)
        
#初始化权值和偏置
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

#卷积池化计算
conv2d_1 = conv2d(x_image,W_conv1) + b_conv1
h_conv1 = tf.nn.relu(conv2d_1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv1_drop = tf.nn.dropout(h_pool1,keep_prob_1)
        
#Conv2&Pool2 
#卷积池化计算
W_conv2 = weight_variable([5,5,32,64])  
b_conv2 = bias_variable([64])

#卷积计算
conv2d_2 = conv2d(h_conv1_drop,W_conv2) + b_conv2
h_conv2 = tf.nn.relu(conv2d_2)
h_pool2 = max_pool_2x2(h_conv2)
h_conv2_drop = tf.nn.dropout(h_pool2,keep_prob_1)
'''        
#Conv3&Pool3          
#初始化权值和偏置
W_conv3 = weight_variable([5,5,64,64])  
b_conv3 = bias_variable([64])
#卷积池化计算
conv2d_3 = conv2d(h_conv2_drop,W_conv3) + b_conv3
h_conv3 = tf.nn.relu(conv2d_3)
h_pool3 = max_pool_2x2(h_conv3)
h_conv3_drop = tf.nn.dropout(h_pool3,keep_prob_1)
 '''       
#Full1
#初始化权值 
W_fc1 = weight_variable([16*16*64,1024])
b_fc1 = bias_variable([1024])

#flat(扁平)
h_pool2_flat = tf.reshape(h_conv2_drop,[-1,16*16*64])
#全连接计算  
wx_plus_b1 = tf.matmul(h_pool2_flat,W_fc1) + b_fc1
h_fc1 = tf.nn.relu(wx_plus_b1)

#dropout
keep_prob_2 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob_2)

#输出
W_fc2 = weight_variable([1024,10]) 
b_fc2 = bias_variable([10])  
prediction= tf.matmul(h_fc1_drop,W_fc2) + b_fc2


#交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)
    
#使用AdamOptimizer进行优化
decay_rate = 0.5
decay_step = 10
lr=0.01

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

#求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        
#合并所有的summary
merged = tf.summary.merge_all()

saver=tf.train.Saver()

#%% training & save
global_steps = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    for i in range(global_steps):
        #训练模型
        tf.train.exponential_decay(lr, global_steps,decay_step, decay_rate, staircase=False)
        batch_xs,batch_ys =  get_batch(train,batch_size)
        summary,loss,_=sess.run([merged,cross_entropy,train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob_1:0.5,keep_prob_2:0.75})
        
        train_writer.add_summary(summary,i)
        va_acc = sess.run(accuracy,feed_dict={x:x_va,y:y_va,keep_prob_1:1.0,keep_prob_2:1.0})
        print ("Iter " + str(i)  +', loss:'+ str(loss) +",  validation  Accuracy= " + str(va_acc))
        if va_acc > 0.98:
                tr_acc = sess.run(accuracy,feed_dict={x:x_tr, y:y_tr,keep_prob_1:1.0,keep_prob_2:1.0})
                saver.save(sess, './model/face_recognit_model')
                print('accuracy higer than 0.98, exited!')
                print ("Iter " + str(i)  +', loss:'+ str(loss) +",  validation  Accuracy= " + str(va_acc),", training Accuracy= ",str(tr_acc))
                sys.exit(0)
    saver.save(sess,'./model/face_recognit_model')

#%%
#模型载入
#导入测试集数据
'''
x_te=np.load('./data/x_te.npy')
y_te=np.load('./data/y_te.npy')    
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'face_recognit_model')
    pre,te_acc = sess.run([prediction,accuracy],feed_dict={x:x_te, y_:y_te,keep_prob_1:1.0,keep_prob_2:1.0})
    pre=tf.cast(tf.argmax(pre,1))
    np.save(pre,'./prediction_labels')
    print('test Accuracy: %f'%(te_acc))

'''


