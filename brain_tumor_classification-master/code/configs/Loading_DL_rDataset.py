#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.utils.data as data


# In[3]:


class Loading(data.Dataset):
    """
    读取dataset和labelset并返回Dataset类
    """
    def __init__(self, dataset_list,labelset_list):
        self.dataset_list=dataset_list
        self.labelset_list=labelset_list
        
    def __getitem__(self,idx):
        dataset=np.array(self.dataset_list[idx])
        labelset=np.array(self.labelset_list[idx])
        dataset=dataset.astype(np.float32)
        labelset=labelset.astype(np.float32)
        dataset=torch.from_numpy(dataset)
        dataset.requires_grad = True
        labelset=torch.from_numpy(labelset)
        return dataset,labelset

    def __len__(self):
        return len(self.dataset_list)

