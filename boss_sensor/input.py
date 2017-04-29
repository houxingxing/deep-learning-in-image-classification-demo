# -*- coding: utf-8 -*-
import numpy as np
import cv2,os

def rescale_image(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(64,64))
    return img

def read_dir(path):
    dir=os.listdir(path)
    n=len(dir)
    data=np.empty((n,64,64,3),dtype='int8')
    i=0
    for d in dir:
        ph=path+"/"+d
        img=rescale_image(ph)
        data[i,:,:,:]=img[:,:,:]
        i+=1
        print(ph)
        print(img.shape)
    return data


def extract_data():
    path1='../Image/boss'
    data1=read_dir(path1)
    label1=np.ones((data1.shape[0],1),dtype='int8')

    path2='../Image/other'
    data2=read_dir(path2)
    label2=np.zeros((data2.shape[0],1),dtype='int8')
    traindata=np.vstack([data1,data2])
    trainlabel=np.vstack([label1,label2])
    traindata=traindata.reshape((traindata.shape[0],64*64*3))
    # np.savetxt("traindata.csv",traindata,delimiter=',',fmt='%d')
    # np.savetxt("trainlabel.csv",trainlabel,delimiter=',',fmt='%d')
    # print(traindata.shape,trainlabel.shape)
    return traindata,trainlabel


