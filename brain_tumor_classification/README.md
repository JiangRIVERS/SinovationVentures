# Brain_Tumor_Classification

## Content
+ code:

    All codes for this project
+ doc

    All documents for this project

## Structure
```
code/
  |--configs/: Data processing and some configs
  |    |--Generate_DL.py: Data processing module
  |    |--Loading_DL_rDataset.py: Convert dataset and labelset into torch.utils.data.Dataset in order to let the model loading dataset and corresponding labelset
  |    |--Setting.py: Default argparse storage module
  |--model/: model for train and val
  |    |--Dropout_Pretrained_ResNet.py: Pretrained ResNet with dropout layer for Ki67.py and AG_GBM.py
  |    |--Dropout_ResNet18.py: ResNet with dropout layer for Ki67.py and AG_GBM.py
  |    |--Pretrained_ResNet.py: Pretrained ResNet without dropout layer for IDH.py, HGG_LGG.py and LGG_GBM.py
  |    |--ThreeDResNet.py: ResNet without dropout layer for IDH.py, HGG_LGG.py and LGG_GBM.py
  |--pretrain/: Pre-trained path storage module
  |    |--resnet_18.pth: Pretrain ResNet18 parameters(大于100MB，无法同步到github上，如果需要请自行搜索MedicalNet的github，里面README中有下载地址)
  |--AG_GBM.py: train and val module for AG and GBM dataset
  |--HGG_LGG.py: train and val module for HGG and LGG dataset
  |--IDH.py: train and val module for IDH dataset
  |--Ki67.py: train and val module for Ki67 dataset
  |--LGG_GBM.py: train and val module for LGG and GBM dataset
  |--Evaluation.py: Evaluation function

```

```
doc/
 |--graph/:
 |    |--3DResNet18.graffle
 |    |--3DResNet18.png
 |    |--Ourmodel.graffle
 |    |--Ourmodel.png
 |    |--basic block.graffle
 |    |--basic block.png
 |    |--flowchart.graffle
 |    |--flowchart.png
 |    |--fastai_exp.png: Loss and lr curve utilizing fastai
 |    |--fastai_exp_result.png: Loss in each epoch utilizing fastai
 |--16_channel.md: 16 channel(data augment) experiment recording document
 |--multi.md: Multi label experiment recording document
 |--Conbined PDF.pdf
 |--Fastai实验.md: Fastai experiment recording document
 |--IDH1、IDH2基因突变在肿瘤中的作用.pdf
 |--[2020.1.18]钱天翼结果分析总结.docx
 |--五折实验结果修改版.md
 |--五折实验结果修改版.pdf
 |--五折实验结果实验版.md
 |--五折实验结果实验版.pdf
 |--参考论文.md
 |--脑瘤分类技术文档【2020.2.5】.docx
```

