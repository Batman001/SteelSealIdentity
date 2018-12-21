# -*- coding:utf-8 -*-

'''
CNN模型训练数据准备
'''


import os
import cv2
import numpy as np


class DataSet(object):
    '''
    定义一个数据集类，用来生成训练集和测试集数据
    Args:
        data_aug: False for valid/testing.
        shuffle: true for training, False for valid/test.
    '''

    def __init__(self, root_dir, dataset, sub_set, batch_size, n_label,
                 data_aug=False, shuffle=True):
        np.random.seed(0)
        self.data_dir = os.path.join(root_dir, dataset, sub_set)  # 进入label目录
        self.batch_size = batch_size  # batch_size=64
        self.n_label = n_label  # n_label=65
        self.data_aug = data_aug  # 是否数据扩增，否
        self.shuffle = shuffle  # 是否打算训练集顺序，训练集打乱，验证集和测试集不打乱
        self.xs, self.ys = self.load_data()  # 获得训练集和label
        self._num_examples = len(self.xs)  # 训练集样本数
        self.init_epoch()

    def load_data(self):  # 遍历图片生成训练集列表
        '''Fetch all data into a list'''
        '''TODO: 1. You may make it more memory efficient if there is a OOM problem on
        you machine. 2. You may use data augmentation tricks.'''
        xs = []
        ys = []
        label_dirs = os.listdir(self.data_dir)  # 返回分类列表
        label_dirs.sort()
        for _label_dir in label_dirs:
            # print (label_dirs)
            #print('loaded {}'.format(_label_dir))
            category = int(_label_dir[2:])  # 将label提取出来
            label = np.zeros(self.n_label)
            label[category] = 1  # 将label用one-hot向量表示
            imgs_name = os.listdir(os.path.join(self.data_dir, _label_dir))  # 返回某一类别下的img列表
            imgs_name.sort()
            for img_name in imgs_name:
                # print (img_name)
                im_ar = cv2.imread(os.path.join(self.data_dir, _label_dir, img_name))  # 读取img
                im_ar = cv2.cvtColor(im_ar, cv2.COLOR_BGR2RGB)  # 将BGE彩色图像转换为RGB彩色图像
                im_ar = np.asarray(im_ar)  # 转化为ndarray
                im_ar = self.preprocess(im_ar)  # 图片前处理，调整大小为224*224并进行归一化
                xs.append(im_ar)  # 存入x列表
                ys.append(label)  # 存入label列表
        return xs, ys

    def preprocess(self, im_ar):  # 调整图片尺寸和像素强度
        '''Resize raw image to a fixed size, and scale the pixel intensities.'''
        '''TODO: you may add data augmentation methods.'''
        im_ar = cv2.resize(im_ar, (48, 48))
        im_ar = im_ar / 255.0
        return im_ar

    def next_batch(self):  #生成下一个batch的训练数据
        '''Fetch the next batch of images and labels.'''
        if not self.has_next_batch():
            self.init_epoch()#return None
        print(self.cur_index)
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            x_batch.append(self.xs[self.indices[self.cur_index + i]])
            y_batch.append(self.ys[self.indices[self.cur_index + i]])
        self.cur_index += self.batch_size
        return np.asarray(x_batch), np.asarray(y_batch)

    def has_next_batch(self):  #判断是否还有下一个batch
        '''Call this function before fetching the next batch.
        If no batch left, a training epoch is over.'''
        start = self.cur_index
        end = self.batch_size + start
        if end > self._num_examples:
            return False
        else:
            return True

    def init_epoch(self):  #所有数据参与过一次训练后，打乱顺序生成新的训练数据
        '''Make sure you would shuffle the training set before the next epoch.
        e.g. if not train_set.has_next_batch(): train_set.init_epoch()'''
        self.cur_index = 0
        self.indices = np.arange(self._num_examples)
        if self.shuffle:
            np.random.shuffle(self.indices)