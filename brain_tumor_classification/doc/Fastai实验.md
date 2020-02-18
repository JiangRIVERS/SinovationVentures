# 使用fastai寻找lr从而优化模型结果

在论文中说CLR相对于自适应optimizer节省了一部分计算量，这个计算量来自于自适应optimizer需要根据之前的值决定当前lr，而CLR不需要，因而CLR可以视为自适应optimizer的竞争者。

但是在实际使用过程中，还是存在一些问题。

1. 说明文档和当前fastai的版本不对应，导致文档中某些类中的某些函数无法使用。

2. 模型过于集成化，适合于新手练手或者寻找初始学习率，和用pytorch搭建并训练模型的过程暂时不可同日而语。（见 https://github.com/JiangRIVERS/Some_tricks_when_using_Pytorch/blob/master/使用Pytorch训练模型时踩的坑.md 中的第五条）

下面是使用fastai进行IDH训练的结果。fastai应该是采用了某些节省显存的操作，之前运行模型使12G的GPU爆显存，通过checkpoint操作使GPU显存降到4GB，而使用fastai时，不使用checkpoint操作（fastai集成化导致checkpoint操作本身不可使用）GPU显存占用降到2GB。

<img src="/Users/jiangmingda/PycharmProjects/Sinovation_Ventures/Project1/fastai_exp.png" width="50%">

根据lr_loss曲线选择学习率lr=1e-3，之前自适应optimizer选取lr=1e-5

但是结果一般

<img src="/Users/jiangmingda/PycharmProjects/Sinovation_Ventures/Project1/fastai_exp_result.png" width="50%">

所以归根到底还是模型和数据的原因，fastai可能会帮助选取一个较为合适的lr，而且集成化的模型操作适合新手去了解并跑简单的模型，但是并不能真正意义上做到较大的提升且有些操作在fastai上较为不便。
