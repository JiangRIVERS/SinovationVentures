#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.metrics as metrics


# In[ ]:


class Evaluation:
    '''输入需要是一维numpy.array，代表每个病例的预测类别'''
    def __init__(self,y_true,y_pred):
        self.y_true=y_true
        self.y_pred=y_pred
        confusion_matrix=metrics.confusion_matrix(y_true,y_pred)
        self.TP=confusion_matrix[1][1]
        self.FP=confusion_matrix[0][1]
        self.FN=confusion_matrix[1][0]
        self.TN=confusion_matrix[0][0]


        
    def ACC(self):
        #准确率
        #所有中完全分对的
        #(TP+TN)/(TP+FN+FP+TN)
        return metrics.accuracy_score(self.y_true,self.y_pred)
    
    def SN(self):
        #敏感度(召回率)
        #漏诊，所有正类中预测为正类的
        #TP/(TP+FN)
        return metrics.recall_score(self.y_true,self.y_pred)
    
    def SP(self):
        #特异性
        #误诊，所有负类中预测为负类的
        #TN/(TN+FP)
        return self.TN/(self.TN+self.FP)
    
    def PPV(self):
        #阳性预测率(精确率)
        #被分类为正类的中分对为正类的
        #TP/(TP+FP)
        return metrics.precision_score(self.y_true,self.y_pred)
    
    def NPV(self):
        #阴性预测率
        #被分类为负类的中分对为负类的
        #TN/(TN+FN)
        return self.TN/(self.TN+self.FN)
    
    def AUC(self):
        #ROC曲线下面积
        return metrics.roc_auc_score(self.y_true,self.y_pred)
        
        


# In[2]:


#import numpy as np
#y_pred=np.array([1,0,1,1,1])
#y_true=np.array([1,0,1,1,0])
#confusion_matrix=metrics.confusion_matrix(y_true,y_pred)


# In[12]:


#confusion_matrix


# In[13]:


#metrics.recall_score(y_true,y_pred)


# In[14]:


#metrics.accuracy_score(y_true,y_pred)


# In[16]:


#metrics.roc_auc_score(y_true,y_pred)


# In[ ]:




