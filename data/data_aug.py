'''
keras提供的数据增广函数
'''


import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras import backend as K
import glob
import keras
import cv2
import numpy as np
import os


def image_create():
    datagen = image.ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=45.0,  #随机旋转角度
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=10.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=K.image_data_format()
    )


    gen_data = datagen.flow_from_directory("./data_demo",
                                           batch_size=1,
                                           shuffle=False,
                                           save_to_dir="./data_demo_aug",
                                           save_prefix='',
                                           target_size=(48,48))
    for i in range(200):
        gen_data.next()

image_create()


