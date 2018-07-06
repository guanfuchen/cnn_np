# cnn_np

本仓库主要使用numpy来实现深度学习中常用的网络结构，复习深度学习相关知识。

使用构造的网络模块搭建LeNet-5（结构参考Caffe实现）训练手写数字识别。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_4/lenet_caffe.png)

---
## 卷积

卷积模块实现时，使用标准模块测试输入输出。

卷积前向传播较为简单，只需要将权值转换为列向量，图像转换为列向量，然后按照矩阵乘法同时加上偏置在转换为图像即可。

卷积反向传播较难，首先通过误差计算权值和偏置的梯度，然后计算该误差下一传播过程中的误差，该计算中注意误差需要通过pad然后转换为列向量，和权重上下翻转后相乘即可。

[pytorch卷积实现conv.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py)

[在 Caffe 中如何计算卷积？](https://www.zhihu.com/question/28385679/answer/44297845) 贾扬清讲解caffe中如何计算卷积。

[Convolutional Neural Networks backpropagation: from intuition to derivation](https://grzegorzgwardys.wordpress.com/2016/04/22/8/)


---
# 池化层

本仓库实现了MaxPool2d。

[Convnet: Implementing Maxpool Layer with Numpy](https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/)

---
## 全连接层

全连接层

[linear.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py)


---
## Softmax层

Softmax层当前神经元的输出对当前的神经元的导数之和$-a_i*a_j$以及$a_i*(1-a_i)$。

[softmax函数与交叉熵的反向梯度传导](https://blog.csdn.net/fireflychh/article/details/73794270) 本文介绍了softmax层的反向传播。


---
## 激活函数层

[activation.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py) pytorch中激活函数设计参考。

---
## 损失函数层

CrossEntropy层使用了one-hot编码的target。

[pytorch loss function 总结](https://blog.csdn.net/zhangxb35/article/details/72464152) 其中介绍了pytorch中各个损失函数的关系，本仓库实现CrossEntropy包含LogSoftmax。

[loss.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py) pytorch中相应的loss设计参考。

[One Hot Encoding using numpy](https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy)

[numpy之one-hot](https://blog.csdn.net/he_wen_jie/article/details/78190517)

---
## 实现日志

- [x] Conv2d
- [x] Pool2d
- [x] Linear
- [x] Softmax
- [x] CrossEntropy
- [x] ReLU
- [ ] LReLU

---
## 参考链接

[手把手带你用Numpy实现CNN <零>](https://zhuanlan.zhihu.com/p/33773140) 作者也是基于这个思路讲解项目思路。

[CNN-Numpy](https://github.com/wuziheng/CNN-Numpy) 代码可以参考。

[贾扬清分享_深度学习框架caffe](http://www.datakit.cn/blog/2015/06/12/online_meet_up_with_yangqing_jia.html) 贾扬清关于caffe的一些分享。

[Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/) cs231n中关于卷积的教程。

[Implementing convolution as a matrix multiplication](https://buptldy.github.io/2016/10/01/2016-10-01-im2col/) 实现卷积操作。

[lenet.prototxt](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt) Caffe中关于LeNet-5的相关实现。

[Netscope](http://ethereon.github.io/netscope/quickstart.html) Netscope一款开源的网络可视化框架（支持caffe）。

[pytorch mnist code](https://github.com/pytorch/examples/blob/master/mnist/main.py) pytorch中mnist代码参考。

[pytorch modules](https://github.com/pytorch/pytorch/tree/master/torch/nn/modules) pytorch中深度学习模块实现列表。

[NumPyCNN](https://github.com/ahmedfgad/NumPyCNN) 使用np实现了卷积层、ReLU和最大池化层。
