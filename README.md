# PR1-DenseNet

This Repo Contains the Implementation of DenseNet

DenseNet Class Takes Following arguments:

growth_rate : you have to pass the growth rate by which you want to train the model.
input_shape: shape of the input image.
nclasses: the number of target classes you have in your dataset.
compression :  this argument control the amount of downsamlping that is done at transition layer (should conatin value between 0 & 1).
bottleleck_flag (bool) : To be used to activate bottelneck layers
variant:  as there are multiple variant (this repo support following variant Densenet-121, Densenet-169, Densenet-201, Densenet-161)


the densenet.py contains the source code.

USAGE:
model = Densenet(12,224,10, True)




Paper Link : https://arxiv.org/pdf/1608.06993.pdf