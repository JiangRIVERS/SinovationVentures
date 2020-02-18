#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import ThreeDResNet


def generate_model(opt):
    """
    Param:parse_opts from setting
    Return: A pretained resnet based on the path:parse_opts.pretain_path
    """
    model = ThreeDResNet.resnet18(
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda)
    net_dict = model.state_dict()
    # load pretrain
    print('loading pretrained model {}'.format(opt.pretrain_path))
    if opt.no_cuda:
        pretrain = torch.load(opt.pretrain_path,map_location='cpu')
    else:pretrain = torch.load(opt.pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    #固定除fc层外参数
    #for param in model.parameters():
     #   param.requires_grad = False
    #for param in model.layer4.parameters():
     #   param.requires_grad=True
    #for param in model.fc.parameters():
      #  param.requires_grad = True
    

    return model


# In[ ]:




