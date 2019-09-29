# encoding: utf-8
"""
@author: lee
@time: 2019/9/27 22:40
@file: main.py
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset


def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。

        参数：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

        返回：
            w  - 维度为（dim，1）的初始化向量。
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros(shape=(dim, 1))
    b = 0
    #使用断言来确保我要的数据是正确的
    assert(w.shape == (dim, 1)) #w的维度是(dim,1)
    assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int
    return (w, b)


def sigmoid(z):
    """
    参数：
        z  - 任何大小的标量或numpy数组。

    返回：
        s  -  sigmoid（z）
    """
    s = 1 / (1 + np.exp(-z))
    return s


if __name__ == '__main__':
    # train_set_x_orig 209,64,64,3
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    index = 25
    plt.imshow(train_set_x_orig[2])
    # plt.show()
    # 将训练集的维度降低并转置。train_set_x_orig.shape[0]为209，-1为根据其余维来推断，(209,12288),T转置为(12288,209)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    # 将测试集的维度降低并转置。
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    '''
    训练集降维最后的维度： (12288, 209)
    训练集_标签的维数 : (1, 209)
    测试集降维之后的维度: (12288, 50)
    测试集_标签的维数 : (1, 50)
    '''
    # 像素值实际上是从0到255范围内
    # 机器学习中一个常见的预处理步骤是对数据集进行居中和标准化
    # 但对于图片数据集，它更简单，更方便，几乎可以将数据集的每一行除以255（像素通道的最大值）
    # 因为在RGB中不存在比255大的数据，所以我们可以放心的除以255，让标准化的数据位于[0,1]之间
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255
    print()
