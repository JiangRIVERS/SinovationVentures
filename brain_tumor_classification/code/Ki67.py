#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:


import torch
import numpy as np
from Generate_DL import data_path,generate_datalabel
import random
from Loading_DL_rDataset import Loading
from Setting import parse_opts
import torch.utils.data as unilsdata
from Dropout_Pretrained_ResNet import generate_model
import torch.optim as optim
import numpy as np
import torch.nn as nn
from Evaluation import Evaluation
from torch.utils.checkpoint import checkpoint


# In[4]:


#from imp import reload
#reload(xxx)


# In[5]:


sets=parse_opts()


# In[6]:


torch.manual_seed(sets.manual_seed)
if not sets.no_cuda:
    #为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed_all(sets.manual_seed)


# # 数据处理
# + 读取数据
# + 划分训练集和测试集
# + 将training、testing数据打包为DataLoader,数据处理为tensor,data.requires_grad=True.dtype==float32

# In[7]:


#IDH_tub病例，对应label设为'tub',输出data尺寸为4*32*240*240
#data.dtype==float32,label.dtype==int64
tub_path=data_path('/Ki67/ki67_01')
tub_dataset=[]
tub_labelset=[]
for idx,tub_patient in enumerate(tub_path):
    class_=generate_datalabel(tub_patient,'tub')
    #每个tub_patient为一个病人的flair、seg、t1、t1ce、t2列表（按顺序）:
    for modal in range(len(tub_patient)-1):
        if modal==0:
            data=class_.normalize(modal=modal+1)[np.newaxis,:]
        else:
            data=np.concatenate((data,class_.normalize(modal=modal+1)[np.newaxis,:]),axis=0)
    label=class_.generate_labelset()
    tub_dataset.append(data)
    tub_labelset.append(label)


# In[8]:


#IDH_yes病例，对应label设为'yes'，其中第188个病例最大mask对应idx为7，因而取 0-31 idx slices
yes_path=data_path('/Ki67/ki67_23')
yes_dataset=[]
yes_labelset=[]
for idx,yes_patient in enumerate(yes_path):
    class_=generate_datalabel(yes_patient,'yes')
    for modal in range(len(yes_patient)-1):
        if modal==0:
            data=class_.normalize(modal=modal+1)[np.newaxis,:]
        else:
            data=np.concatenate((data,class_.normalize(modal=modal+1)[np.newaxis,:]),axis=0)
    label=class_.generate_labelset()
    yes_dataset.append(data)
    yes_labelset.append(label) 


# In[9]:


#设置随机数
random.seed(2020)

tub_shuf_seq=list(range(len(tub_dataset)))
random.shuffle(tub_shuf_seq)
tub_len=len(tub_shuf_seq)//5

yes_shuf_seq=list(range(len(yes_dataset)))
random.shuffle(yes_shuf_seq)
yes_len=len(yes_shuf_seq)//5


#5-fold
tub_part_1=tub_shuf_seq[:tub_len]
tub_part_2=tub_shuf_seq[tub_len:2*tub_len]
tub_part_3=tub_shuf_seq[2*tub_len:3*tub_len]
tub_part_4=tub_shuf_seq[3*tub_len:4*tub_len]
tub_part_5=tub_shuf_seq[4*tub_len:]

yes_part_1=yes_shuf_seq[:yes_len]
yes_part_2=yes_shuf_seq[yes_len:2*yes_len]
yes_part_3=yes_shuf_seq[2*yes_len:3*yes_len]
yes_part_4=yes_shuf_seq[3*yes_len:4*yes_len]
yes_part_5=yes_shuf_seq[4*yes_len:]


#fold3
tub_train=tub_part_1
tub_train.extend(tub_part_2)
tub_train.extend(tub_part_4)
tub_train.extend(tub_part_5)
tub_test=tub_part_3

yes_train=yes_part_1
yes_train.extend(yes_part_2)
yes_train.extend(yes_part_4)
yes_train.extend(yes_part_5)
yes_test=yes_part_3




# In[10]:


#分别对tub、yes病例取training vs testing==4:1,保证数据分布
tub_shuf_training_dataset=[]
tub_shuf_training_labelset=[]
tub_shuf_testing_dataset=[]
tub_shuf_testing_labelset=[]

yes_shuf_training_dataset=[]
yes_shuf_training_labelset=[]
yes_shuf_testing_dataset=[]
yes_shuf_testing_labelset=[]

for idx in tub_train:
    tub_shuf_training_dataset.append(tub_dataset[idx])
    tub_shuf_training_labelset.append(tub_labelset[idx])    
print('tub_shuf_trainging complete.')

for idx in yes_train:
    yes_shuf_training_dataset.append(yes_dataset[idx])
    yes_shuf_training_labelset.append(yes_labelset[idx])
print('yes_shuf_training complete.')
    
for idx in tub_test:
    tub_shuf_testing_dataset.append(tub_dataset[idx])
    tub_shuf_testing_labelset.append(tub_labelset[idx])
print('tub_shuf_testing complete.')

for idx in yes_test:
    yes_shuf_testing_dataset.append(yes_dataset[idx])
    yes_shuf_testing_labelset.append(yes_labelset[idx])
print('yes_shuf_testing complete.')

dataset_training=tub_shuf_training_dataset
dataset_training.extend(yes_shuf_training_dataset)
labelset_training=tub_shuf_training_labelset
labelset_training.extend(yes_shuf_training_labelset)

dataset_testing=tub_shuf_testing_dataset
dataset_testing.extend(yes_shuf_testing_dataset)
labelset_testing=tub_shuf_testing_labelset
labelset_testing.extend(yes_shuf_testing_labelset)
print('len_of_trainingdataset:{},len_of_testdataset:{}'.format(len(dataset_training),len(dataset_testing)))

# In[11]:


#生成DataLoader对象
kwargs={'num_workers': 0, 'pin_memory': True} if not sets.no_cuda else {}
training_loader=unilsdata.DataLoader(Loading(
    dataset_training,labelset_training),batch_size=4,shuffle=True,drop_last=True,**kwargs)
testing_loader=unilsdata.DataLoader(Loading(
    dataset_testing,labelset_testing),batch_size=4,shuffle=True,drop_last=True,**kwargs)


# # 模型

# In[12]:


model=generate_model(sets)
if not sets.no_cuda:
    model = nn.DataParallel(model)
    model.cuda()


# In[13]:


optimizer=optim.Adam(model.parameters(),lr=3e-4,betas=sets.betas,weight_decay=3e-4)


# In[14]:


criterion=nn.BCELoss()


# # Training && Testing

# In[15]:


Softmax= nn.Softmax(dim=1)
def train(epoch):
    model.train()
    train_loss=0.
    pred=[]
    true=[]
    for idx,(data,label) in enumerate(training_loader):
        if not sets.no_cuda:
            data,label=data.cuda(),label.cuda()
        optimizer.zero_grad()
        
        output=checkpoint(model.module.forward_one,data)
        output=checkpoint(model.module.forward_two,output)
        #output=model(data)
        
        output_softmax=Softmax(output)
        loss=criterion(output_softmax,label)
        loss.backward()
        optimizer.step()
            
        train_loss+=loss
        
        _,predict=torch.max(output_softmax,1)
        _,truth=torch.max(label,1)
        pred.append(predict.cpu().numpy()[0])
        true.append(truth.cpu().numpy()[0])
            
    #evaluation
    evaluation=Evaluation(np.array(true),np.array(pred))
    acc=evaluation.ACC()#准确率
    sn=evaluation.SN()#敏感度(召回率)
    sp=evaluation.SP()#特异性
    ppv=evaluation.PPV()#阳性预测率(准确率)
    npv=evaluation.NPV()#阴性预测率
    auc=evaluation.AUC()#ROC曲线下面积
    
    print('--------------------------------------------') 
    print('Ki67_Fold3:Train Epoch:{}-->Loss:{:6f}-->ACC:{:6f}-->SN:{:6f}-->SP:{:6f}-->PPV:{:6f}-->NPV:{:6f}-->AUC:{:6f}'
                .format(epoch,train_loss/idx,acc,sn,sp,ppv,npv,auc))          


# In[16]:


def test(epoch):
    model.eval()
    test_loss=0.
    pred=[]
    true=[]
    for idx,(data,label) in enumerate(testing_loader):
        if not sets.no_cuda:
            data,label=data.cuda(),label.cuda()
            
        output=checkpoint(model.module.forward_one,data)
        output=checkpoint(model.module.forward_two,output)
        
        #output=model(data)
        output_softmax=Softmax(output)
        loss=criterion(output_softmax,label)
            
        test_loss+=loss
        
        _,predict=torch.max(output_softmax,1)
        _,truth=torch.max(label,1)
        pred.append(predict.cpu().numpy()[0])
        true.append(truth.cpu().numpy()[0])
            
    #evaluation
    evaluation=Evaluation(np.array(true),np.array(pred))
    acc=evaluation.ACC()
    sn=evaluation.SN()
    sp=evaluation.SP()
    ppv=evaluation.PPV()
    npv=evaluation.NPV()
    auc=evaluation.AUC()
    
    print('--------------------------------------------') 
    print('Ki67_Fold3:Test Epoch:{}-->Loss:{:6f}-->ACC:{:6f}-->SN:{:6f}-->SP:{:6f}-->PPV:{:6f}-->NPV:{:6f}-->AUC:{:6f}'
                .format(epoch,test_loss/idx,acc,sn,sp,ppv,npv,auc))          


# # IDH

# In[19]:


#i=1
for epoch in range(1,41):
    train(epoch)
    test(epoch)
    #torch.save({
     #   'epoch': epoch,
      #  'model_state_dict': model.module.state_dict(),
       # 'optimizer_state_dict': optimizer.state_dict(),
        #'loss':nn.BCELoss(),
        #}, '1e-3Allparams'+str(i)+'.pkl')
    #i+=1


# # Ki67

# In[17]:


#i=1
#for epoch in range(1,51):
 #   train(epoch)
 #   test(epoch)
    #torch.save({
     #   'epoch': epoch,
      #  'model_state_dict': model.module.state_dict(),
       # 'optimizer_state_dict': optimizer.state_dict(),
        #'loss':nn.BCELoss(),
        #}, 'Ki67_1e-5Allparams'+str(i)+'.pkl')
    #i+=1


# In[17]:


#i=1
#for epoch in range(1,21):
 #   train(epoch)
#    test(epoch)
    #torch.save({
     #   'epoch': epoch,
      #  'model_state_dict': model.module.state_dict(),
       # 'optimizer_state_dict': optimizer.state_dict(),
        #'loss':nn.BCELoss(),
        #}, 'Ki67_1e-5Allparams'+str(i)+'.pkl')
    #i+=1


# In[17]:


#i=1
#for epoch in range(1,31):
#    train(epoch)
 #   test(epoch)
    #torch.save({
     #   'epoch': epoch,
      #  'model_state_dict': model.module.state_dict(),
       # 'optimizer_state_dict': optimizer.state_dict(),
        #'loss':nn.BCELoss(),
        #}, 'Ki67_1e-5Allparams'+str(i)+'.pkl')
    #i+=1


# In[18]:


#i=31
#for epoch in range(31,51):
  #  train(epoch)
  #  test(epoch)
    #torch.save({
     #   'epoch': epoch,
      #  'model_state_dict': model.module.state_dict(),
       # 'optimizer_state_dict': optimizer.state_dict(),
        #'loss':nn.BCELoss(),
        #}, 'Ki67_1e-5Allparams'+str(i)+'.pkl')
    #i+=1


# In[ ]:


#i=51
#for epoch in range(51,101):
 #   train(epoch)
  #  test(epoch)
    #torch.save({
     #   'epoch': epoch,
      #  'model_state_dict': model.module.state_dict(),
       # 'optimizer_state_dict': optimizer.state_dict(),
        #'loss':nn.BCELoss(),
        #}, 'Ki67_1e-5Allparams'+str(i)+'.pkl')
    #i+=1


# In[ ]:


