# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 05:11:56 2018

@author: Administrator
"""

import paddle.v2 as paddle
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
###
#n_total=3450
n_total=3450
n_pass=1
n_buf=100
n_bat_size=50
###
def rgb2gray(rgb):
 
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
###
import datetime
starttime = datetime.datetime.now()
#
path_top='/home/aistudio/work/baidu_star_2018'
path_work='/home/aistudio/work'
#
path_top='/home/vuv/baidu_star_2018'
path_work='/home/vuv'
#
ftar=os.path.join(path_work,'mcnn_pc2.tar')
path=os.path.join(path_top,'image','stage1')
#
path_den50=os.path.join(path,'den50')
path_im=os.path.join(path,'train')
parents = os.listdir(path_den50)
n_dir=len(parents)
n_data=1000
#
#size=150
size=80

size_out=50
dim=size*size
dim_out=size_out*size_out
def train_reader():
    def reader():
        for i in range(n_total):
            #3450 den50  3618 img
            imod=i
            parent=parents[imod]
            fden50=os.path.join(path_den50,parent)
            den50=np.load(fden50)
            fim=os.path.join(path_im,parent[:-4])
            im_ori=cv2.imread(fim)
            im_ori=rgb2gray(im_ori)
            im = cv2.resize(im_ori,(size, size),interpolation=cv2.INTER_AREA)
            #cc=np.ones([size,size])
            #dd=np.random.rand(size_out,size_out)
	    yield [im],[den50]
    return reader

#初始化Paddle
paddle.init(use_gpu=False)
 #配置训练网络
img = paddle.layer.data(name='img', type=paddle.data_type.dense_vector(dim))
#conv1 = paddle.layer.img_conv(input=img,filter_size=7,num_channels=1,num_filters=3,stride=2)
#conv2 = paddle.layer.img_conv(input=conv1,filter_size=7,num_filters=32,stride=2)
#conv3 = paddle.layer.img_conv(input=conv2,filter_size=5,num_filters=32,stride=2)
###simple MCNN
#conv3 = paddle.layer.img_conv(name='conv3',input=img,filter_size=3,num_channels=1,num_filters=6,stride=2,padding=0)
#conv5 = paddle.layer.img_conv(name='conv5',input=img,filter_size=5,num_channels=1,num_filters=5,stride=2,padding=1)
#conv7 = paddle.layer.img_conv(name='conv7',input=img,filter_size=7,num_channels=1,num_filters=4,stride=2,padding=2)
#cat = paddle.layer.concat(name='cat', input=[conv3, conv5, conv7])
###MCNN
conv_stride=1
#ch_base=8 #48 40 32
ch_base=8
ch3=6*ch_base
c31 = paddle.layer.img_conv(name='c31',input=img,padding=0,filter_size=5,num_channels=1,num_filters=int(ch3/2),stride=conv_stride)
c32 = paddle.layer.img_conv(input=c31,padding=0,filter_size=3,num_filters=ch3,stride=conv_stride)
p32=paddle.layer.img_pool(input=c32,pool_size=2,stride=2)
c33 = paddle.layer.img_conv(input=p32,padding=0,filter_size=3,num_filters=int(ch3/2),stride=conv_stride)
p33=paddle.layer.img_pool(input=c33,pool_size=2,stride=2)
c34 = paddle.layer.img_conv(input=p33,padding=0,filter_size=3,num_filters=int(ch3/4),stride=conv_stride)
ch5=5*ch_base
c51=paddle.layer.img_conv(name='c51',input=img,padding=1,filter_size=7,num_channels=1,num_filters=int(ch5/2),stride=conv_stride)
c52=paddle.layer.img_conv(input=c51,padding=1,filter_size=5,num_filters=ch5,stride=conv_stride)
p52=paddle.layer.img_pool(input=c52,pool_size=2,stride=2)
c53=paddle.layer.img_conv(input=p52,padding=1,filter_size=5,num_filters=int(ch5/2),stride=conv_stride)
p53=paddle.layer.img_pool(input=c53,pool_size=2,stride=2)
c54 = paddle.layer.img_conv(input=p53,padding=1,filter_size=5,num_filters=int(ch5/4),stride=conv_stride)
ch7=4*ch_base
c71=paddle.layer.img_conv(name='c71',input=img,padding=2,filter_size=9,num_channels=1,num_filters=int(ch7/2),stride=conv_stride)
c72=paddle.layer.img_conv(input=c71,padding=2,filter_size=7,num_filters=ch7,stride=conv_stride)
p72=paddle.layer.img_pool(input=c72,pool_size=2,stride=2)
c73=paddle.layer.img_conv(input=p72,padding=2,filter_size=7,num_filters=int(ch7/2),stride=conv_stride)
p73=paddle.layer.img_pool(input=c73,pool_size=2,stride=2)
c74 = paddle.layer.img_conv(input=p73,padding=2,filter_size=7,num_filters=int(ch7/4),stride=conv_stride)
###
cat = paddle.layer.concat(name='cat', input=[c34,c54,c74])


###################################################
fc = paddle.layer.fc(input=cat, size=dim_out, act=paddle.activation.Linear())
label = paddle.layer.data(name='label', type= paddle.data_type.dense_vector(dim_out))
cost = paddle.layer.square_error_cost(input=fc, label=label)
#cost = paddle.layer.mse_cost(input=fc, label=nety)

######创建参数parameters###############################
#parameters = paddle.parameters.create(cost)

with open(ftar, 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)
######创建Trainer#######################################
#optimizer = paddle.optimizer.Momentum(momentum=0)
optimizer = paddle.optimizer.Adam()

trainer = paddle.trainer.SGD(cost= cost,
                            parameters=parameters,
                            update_equation=optimizer)

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        with open(ftar, 'w') as f:
            parameters.to_tar(f)
        if event.batch_id % 1 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost)
            flog = open("./1.log", 'a+')
            print>>flog, "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost)
 

# 开始训练
trainer.train(paddle.batch(paddle.reader.shuffle(train_reader(),buf_size=n_buf),batch_size=n_bat_size),num_passes=n_pass,event_handler=event_handler)
###
endtime = datetime.datetime.now()
print('time consumed is')
print (endtime - starttime)
print('after date')
#with open(ftar, 'w') as f:
#    parameters.to_tar(f)
#####test net##########################
def test_reader():
    def reader():
        for i in range(5):
            #3450 den50  3618 img
            imod=i
            parent=parents[imod]
            fden50=os.path.join(path_den50,parent)
            den50=np.load(fden50)
            fim=os.path.join(path_im,parent[:-4])
            im_ori=cv2.imread(fim)
            im_ori=rgb2gray(im_ori)
            im = cv2.resize(im_ori,(size, size),interpolation=cv2.INTER_AREA)
            #cc=np.ones([size,size])
            #dd=np.random.rand(size_out,size_out)
	    yield [im],[den50]
    return reader
test_data = []
test_label = []
test_data_creator = test_reader()
for item in test_data_creator():
    test_data.append((item[0], ))
    test_label.append(item[1])
    #print(item)
probs = paddle.infer(output_layer=fc, parameters=parameters, input=test_data)
for i in xrange(len(probs)):
    #print(test_label[i],"predict",probs[i][0])
    xx=test_data[i][0][0]
    yy=test_label[i][0]
    yp=probs[i]
    yp=yp.reshape([size_out,size_out])
    print(i,"refnum predictnum",sum(sum(yy)),sum(sum(yp)))
i=4
xx=test_data[i][0][0]
yy=test_label[i][0]
yp=probs[i]
yp=yp.reshape([size_out,size_out])
plt.subplot(311)
plt.imshow(xx)
plt.subplot(312)
plt.imshow(yy)
plt.subplot(313)
plt.imshow(yp)
plt.show()
