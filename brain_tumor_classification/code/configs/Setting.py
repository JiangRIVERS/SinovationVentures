#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse


# In[ ]:


def parse_opts():
    parser = argparse.ArgumentParser(description='Pretrain-ResNet18')
    parser.add_argument(
        '--epochs', type=int, default=30, metavar='N',
        help='number of epochs to train.')
    parser.add_argument(
        '--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate.')
    parser.add_argument(
        '--betas', type=float, default=(0.9, 0.999), metavar='Betas',
                        help='Adam betas.')
    parser.add_argument(
        '--pretrain_path',
        default='resnet_18.pth',
        type=str,
        help=
        'Path for pretrained model.'
    )
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (8)')
    parser.add_argument(
        '--resnet_shortcut',
        default='A',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--manual_seed', default=2020, type=int, help='Manually set random seed')
    
    args = parser.parse_args(args=[])

    return args

