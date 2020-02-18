#!/usr/bin/env python
# coding: utf-8


# In[1]:


import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import random


# In[2]:


def read_nii(path):
    image=sitk.ReadImage(path)
    image_arr=sitk.GetArrayFromImage(image)
    return image_arr


# In[3]:


def data_path(filename):
    #产生路径list
    nii_path=[]
    path='/home/jiangmingda/郑大一脑肿瘤分级数据2019-12-12'+filename
    dirs=os.listdir(path)
    
    for file in dirs:
        if file=='.DS_Store':
            continue
        sub_path=path+'/'+file
        sub_dirs=os.listdir(sub_path)
        temp_list=[]
        for sub_file in sub_dirs:
            subsub_path=path+'/'+file+'/'+sub_file
            if 't1' in subsub_path or 't1ce' in subsub_path or 't2' in subsub_path or 'flair' in subsub_path or 'seg' in subsub_path:
                temp_list.append(subsub_path)
        temp_list.sort()
        temp_list[0],temp_list[1]=temp_list[1],temp_list[0]
        nii_path.append(temp_list)
    return nii_path


# In[4]:


class generate_datalabel:
    '''
    params: > path_list:单个病人的t1、t1ce、t2、flair集合
            > label:病人对应的肿瘤label: 1.'tub' 2.'yes'
    '''
    def __init__(self,path_list,label):
        self.label=label
        self.path_list=path_list  
        
    def read_nii(self,path):
        image=sitk.ReadImage(path)
        image=sitk.GetArrayFromImage(image)
        return image
    
    def generate_mask(self):
        #得到mask，原mask的value range from 0-3，0为背景类，1-3均为肿瘤（不同处理方式或状态下的）
        self.mask=self.read_nii(self.path_list[0])
        ones=np.ones(self.mask.shape,dtype=np.uint8)
        self.mask[self.mask>0]=ones[self.mask>0]
        
    
    def generate_idx_accroding2mask(self):
        #得到最大肿瘤位置，并上下各推16张slice，得到initial和end的idx，用于降维
        self.generate_mask()
        list_=[]
        for channel in range(self.mask.shape[0]):
            list_.append(self.mask[channel,:].sum())
        max_channel=list_.index(max(list_))
        self.initial_1=max_channel-8+1
        self.end_1=max_channel+8
        
        self.initial_2=max_channel-1-8+1
        self.end_2=max_channel+8-1
        
        self.initial_3=max_channel-2-8+1
        self.end_3=max_channel+8-2
        
        self.initial_4=max_channel+1-8+1
        self.end_4=max_channel+1+8
        
        self.initial_5=max_channel+2-8+1
        self.end_5=max_channel+2+8
        
        
        if self.initial_1<0:
            self.initial_1=max_channel+random.randint(0,5)
            self.end_1=self.initial+15
        if self.initial_2<0:
            self.initial_2=max_channel+random.randint(0,5)
            self.end_2=self.initial+15
        if self.initial_3<0:
            self.initial_3=max_channel+random.randint(0,5)
            self.end_3=self.initial+15
        if self.initial_4<0:
            self.initial_4=max_channel+random.randint(0,5)
            self.end_4=self.initial+15
        if self.initial_5<0:
            self.initial_5=max_channel+random.randint(0,5)
            self.end_5=self.initial+15
    
    def generate_image_accroding2mask(self,modal):
        #根据最大肿瘤位置，将155*240*240的volunm降维到32*240*240
        #params: modal:模态。1:flair
        #                   2:t1
        #                   3:t1ce
        #                   4:t2
        self.generate_idx_accroding2mask()
        original_image=self.read_nii(self.path_list[modal])
        self.mask_image_1=original_image[self.initial_1:self.end_1+1]
        self.mask_image_2=original_image[self.initial_2:self.end_2+1]
        self.mask_image_3=original_image[self.initial_3:self.end_3+1]
        self.mask_image_4=original_image[self.initial_4:self.end_4+1]
        self.mask_image_5=original_image[self.initial_5:self.end_5+1]
        
    
    def normalize(self,modal):
        self.generate_image_accroding2mask(modal)
        
        def normalization(mask_image):
            volume=mask_image
            pixels = volume[volume > 0]
            mean = pixels.mean()
            std  = pixels.std()
            out = (volume - mean)/std
            out_random = np.zeros(volume.shape)
            out[volume == 0] = out_random[volume == 0]
            return out
        
        return normalization(self.mask_image_1)[np.newaxis,:],\
               normalization(self.mask_image_2)[np.newaxis,:],\
               normalization(self.mask_image_3)[np.newaxis,:],\
               normalization(self.mask_image_4)[np.newaxis,:],\
               normalization(self.mask_image_5)[np.newaxis,:]
    
    def generate_labelset(self):
        if self.label=='tub':
            return np.array([1,0])
        if self.label=='yes':
            return np.array([0,1])
        else:
            raise KeyError('Wrong label name')


# In[5]:


def saving(list_,address):
    saving_=np.array(list_)
    np.save(address+'.npy',saving_)


# # 将data和label分别对应储存在dataset与labelset中，并保存为.npy文件

# In[6]:


if __name__=='__main__':
    #IDH_tub病例，对应label设为'tub'
    tub_path=data_path('/IDH_tub')
    tub_dataset=[]
    tub_labelset=[]
    for idx,tub_patient in enumerate(tub_path):
        print(idx)
         #每个tub_patient为一个病人的flair、seg、t1、t1ce、t2列表（按顺序）:
        for modal in range(len(tub_patient)-1):
            class_=generate_datalabel(tub_patient,'tub')
            data=class_.normalize(modal=modal+1)
            label=class_.generate_labelset()
            tub_dataset.append(data)
            tub_labelset.append(label)
    #保存
    saving(tub_dataset,'./tubdataset')
    saving(tub_labelset,'./tublabelset')
    #IDH_yes病例，对应label设为'yes'，其中第188个病例最大mask对应idx为7，因而取 0-31 idx slices
    yes_path=data_path('/IDH_yes')
    yes_dataset=[]
    yes_labelset=[]
    for idx,yes_patient in enumerate(yes_path):
        print(idx)
        for modal in range(len(yes_patient)-1):
            class_=generate_datalabel(yes_patient,'yes')
            data=class_.normalize(modal=modal+1)
            label=class_.generate_labelset()
            yes_dataset.append(data)
            yes_labelset.append(label) 
    #保存
    saving(yes_dataset,'./yesdataset')
    saving(yes_labelset,'./yeslabelset')

