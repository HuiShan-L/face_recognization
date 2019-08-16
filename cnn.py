# -*- coding: utf-8 -*-
"""
模型构建与测试
@author: xs
"""
import tensorflow as tf
'''
#创建常量
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[3]])
#创建矩阵乘法
product=tf.matmul(m1,m2)
print(product)

#定义一个会话
sess=tf.Session()
#调用sess的run的方法来执行矩阵乘法op
result=sess.run(product)
print(result)
sess.close()  #关闭会话

##方法二
with tf.Session() as sess:
    result=sess.run(product)
    print(result)
'''   

'''
#变量的使用
x=tf.Variable([1,2])   
a=tf.constant([3,3])

#初始化全局变量
init=tf.global_variables_initializer()
#
#增加一个减法op
sub=tf.subtract(x,a)
#加法op
add=tf.add(x,sub)

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
    
#创建一个变量, 初始化为0    
state=tf.Variable(0,name='counter')
#常见op, 作用是state+1 
new_value=tf.add(state,1)  
#赋值op
update=tf.assign(state,new_value) 
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
'''

'''
#fetch and feed
input1=tf.constant(3.0)    
input2=tf.constant(2.0)      
input3=tf.constant(5.0)     
add=tf.add(input2,input3)
mul=tf.multiply(input1,add)    

#fecth 同时运行多个op
with tf.Session() as sess:
    result=sess.run([mul,add])
    print(result)
    
#feed
#初始化占位符
input1=tf.placeholder(tf.float32)    
input2=tf.placeholder(tf.float32) 
output=tf.multiply(input1,input2)

#feed数据以字典形式传入
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
'''


#/*************************************************
#*********************简单案例***************************
'''
#tensorflow 简单使用案例
import numpy as np
x_data=np.random.rand(100)   
y_data=x_data*0.1+0.2

b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

#定义二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法为优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train=optimizer.minimize(loss)

#初始化
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))
'''


#/*************************************************
#*********************回归***************************
'''
import numpy as np
import matplotlib.pyplot as plt
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义palceholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
  
#构建神经网络结构
Weight_L1=tf.Variable(tf.random_normal([1,10]))   
biase_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weight_L1)+biase_L1
L1=tf.nn.tanh(Wx_plus_b_L1)#激活函数

#定义神经网络输出层
Weight_L2=tf.Variable(tf.random_normal([10,1]))
biase_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weight_L2)+biase_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
'''        

#/*************************************************
#*********************手写数字识别***************************
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#载入数据
mnist=input_data.read_data_sets('MNIST_data/',one_hot = True)

#每个批次大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size

#定义两个plcaeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)  #dropout
lr=tf.Variable(0.001,dtype=tf.float32)  #学习率
#创建神经网络
#W=tf.Variable(tf.zeros([784,10]))
#b=tf.Variable(tf.zeros([10]))
W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1=tf.Variable(tf.zeros([500])+0.1)
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

W2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)

#二次代价函数
#loss=tf.reduce_mean(tf.square(y-prediction))#二次代价函数
#交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)  #优化器
train_step=tf.train.AdamOptimizer(lr).minimize(loss)
#初始化变量
init=tf.global_variables_initializer()

#准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(51):
		sess.run(tf.assign(lr,0.001*(0.95**epoch)))
		for batch in range(n_batch): #获取批次
			batch_xs,batch_ys=mnist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
		acc1=sess.run(acc,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
		acc2=sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
		print('train_acc: %f   test_acc: %f'%(acc1,acc2))
'''


#/*************************************************
#*********************优化器***************************
'''
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss) 
train_step=tf.train.AdamOptimizer(1e-3).minimize(loss)
'''

#/*************************************************
#*********************tensorboard***************************

#tensorboard --logdir=C:\Users\xs\Desktop\td\mtcnn_me\logs
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
#参数概要
def varicabile_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) #标准差
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var)) #最小值
        tf.summary.histogram('histogram',var) #直方图
    
        
#命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y_input')


with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        W1=tf.Variable(tf.truncated_normal([784,100],stddev=0.1))
        varicabile_summaries(W1)
    with tf.name_scope('biases'):
        b1=tf.Variable(tf.zeros([100]))
        varicabile_summaries(b1)
    with tf.name_scope('wx_plus_b'):
        L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
        
#每个批次大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size


W2=tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b2=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(L1,W2)+b2)

#损失函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#优化器
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#准确率
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
varicabile_summaries(acc)
#合并所有summary
merged=tf.summary.merge_all
#变量初始化
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs',sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
    train_acc=sess.run(acc,feed_dict={x:mnist.train.images,y:mnist.train.labels})
    test_acc=sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print('Iter: %d  train_acc: %f  test_acc: %f'%(epoch,train_acc,test_acc))
    writer.add_summary(summary,epoch)
'''
   
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
#载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#运行次数
max_steps = 1001
#图片数量
image_num = 3000
#文件路径
DIR = "./"

#定义会话
sess = tf.Session()

#载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

#参数概要
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

#命名空间
with tf.name_scope('input'):
    #这里的none表示第一个维度可以是任意的长度
    x = tf.placeholder(tf.float32,[None,784],name='x-input')
    #正确的标签
    y = tf.placeholder(tf.float32,[None,10],name='y-input')

#显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    #创建一个简单神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):    
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    #交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#把correct_prediction变为float32类型
        tf.summary.scalar('accuracy',accuracy)

#产生metadata文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(image_num):   
        f.write(str(labels[i]) + '\n')        
        
#合并所有的summary
merged = tf.summary.merge_all()   


projector_writer = tf.summary.FileWriter(DIR + 'projector/projector',sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config)

for i in range(max_steps):
    #每个批次100个样本
    batch_xs,batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    
    if i%100 == 0:
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print ("Iter " + str(i) + ", Testing Accuracy= " + str(acc))

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
projector_writer.close()
sess.close()
'''
#/*************************************************
#*********************cnn***************************

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

#参数
batch_size=100
n_batch=mnist.train.num_examples//batch_size


#初始化权值
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
    initial=tf.constant(0,1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #x input tensor for shape [batch,in_height,in_wight, in_channels]
    #W filter / kernel of shape [filter_height,filter_weight,in_channels,out_channel]
    #strides[0]=strides[3]=1, strides[1]代表x方向的步长, strides[2]代表y方向的步长
    #padding: "SAME" ,"VALID"
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1] 窗口大小
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#定义placeholder
x=tf.placeholder(tf.float32,[None,784]) #28*28
y=tf.placeholder(tf.float32,[None,10])

#改变x的格式转为4D的向量[batch,in_height,in_wight, in_channels]
x_image=tf.reshape(x,[-1,28,28,1])

#初始化第一个卷积层
W_conv1=weight_variable([5,5,1,32]) #5*5的采样窗口, 32个卷积核从1个平面抽取特征
b_conv1=bias_variable([32]) #每一个卷积核一个偏置值

#把x_image和权值进行卷积, 在应用于relu激活函数
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#初始化第二个卷积层
W_conv2=weight_variable([5,5,32,64]) #5*5的采样窗口, 64个卷积核从32个窗口抽取特征
b_conv2=bias_variable([64])

#第二次卷积池化
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


#28*28的图片第一次卷积后还是32*28*28, 第一次池化后32*14*14
#第二次卷积后64*14*14, 第二次池化后64*7*7

#初始化第一个全连接层
w_fc1=weight_variable([7*7*64,1024]) #上一卷积层有7*7*64个神经元, 全连接有1024个神经元
b_fc1=bias_variable([1024])

#把池化层2的输出扁平化为1维
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#第一个全连接层输出
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#dropout
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

#计算输出
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用AdamOptimizer进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

#正确率
correct_prediction=tf.equal(tf.argmax(y),tf.argmax(prediction))
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#变量初始化
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        acc1=sess.run(acc,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        acc2=sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print('train_acc: %f   test_acc: %f'%(acc1,acc2))   

