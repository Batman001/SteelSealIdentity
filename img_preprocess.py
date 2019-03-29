# -*- coding:utf-8 -*-
'''
输入一张钢印图片，将钢印分割，返回钢印的4个字符列表
'''
import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression


def img_pro():
    img_name = "F0N7.png"
    img = cv.imread("./data/gangyin_source/"+img_name)
    cimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow('src', img)
    #cv.imshow('gray',cimg)
    cv.waitKey(0)

    #使用霍夫变换检测图像中的圆
    circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT, 1,100,
                              param1=100, param2=28,minRadius=50,maxRadius=75)
    cimg = cv.cvtColor(cimg, cv.COLOR_GRAY2BGR)
    circles = np.uint16(np.around(circles))

    #计算用于仿射变换的坐标点，是写死的，需要改进
    center = []
    for i in circles[0,:]:
        cv.circle(cimg, (i[0],i[1]),i[2],(0,0,255),2)
        cv.circle(cimg,(i[0],i[1]),2,(0,255,0),3)
        center.append((i[0],i[1]))
    p1 = np.array(center[0], np.float32)
    p2 = np.array(center[1], np.float32)
    p3 = np.array(center[2], np.float32)
    co = 0.41  #调整钢印定位点的位置
    cent = (((p2 - p1)*co + p1)[0], ((p2 - p1)*co + p1)[1])
    co_up = 0.35
    up = (((p3 - cent)*co_up + cent)[0], ((p3 - cent)*co_up + cent)[1])
    co_left = 0.55
    left = (((cent - p1)*co_left + p1)[0], ((cent - p1)*co_left + p1)[1])
    co_right = 0.28
    right = (((p2 - cent)*co_right + cent)[0], ((p2 - cent)*co_right + cent)[1])
    co_bot = -0.35
    bot = (((p3 - cent)*co_bot + cent)[0], ((p3 - cent)*co_bot + cent)[1])
    first = (right[0] - cent[0] + up[0], right[1] - cent[1] + up[1])
    second = (right[0] - (first[0] -right[0]),right[1] - (first[1] -right[1]))
    third = (up[0] - (cent[0] - left[0]), up[1] - (cent[1] - left[1]))

    #cv.circle(cimg, cent, 2, (255,0,0),3)
    #cv.circle(cimg, up, 2, (255,0,0),3)
    #cv.circle(cimg, left, 2, (255,0,0),3)
    #cv.circle(cimg, right, 2, (255,0,0),3)
    cv.circle(cimg, first, 2, (255,0,0),3)
    #cv.circle(cimg, bot, 2, (255,0,0),3)
    cv.circle(cimg, second, 2, (255,0,0),3)
    cv.circle(cimg, third, 2, (255,0,0),3)

    cv.imshow('circles', cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()


    #仿射变换，将倾斜图片调正
    p1 = np.array(first)
    p2 = np.array(second)
    p3 = np.array(third)

    pts1 = np.float32([p1,p2,p3])
    w = np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))
    h = np.sqrt(np.square(p1[0]-p3[0]) + np.square(p1[1]-p3[1]))
    pts2 = np.float32([[0,0],[w,0],[0,h]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(img,M,(int(w),int(h)))

    #cv.imwrite(".\GANGYINSRC\GANGYIN\\"+img_name,dst)
    cv.imshow("gangyin",dst)


    #print (dst.shape)
    #将钢印分割为单个字符，取各边中点进行分割
    ims = []
    w1, h1 = dst.shape[:2]
    ims.append(dst[0:int(w1/2), 0:int(h1/2)])
    ims.append(dst[0:int(w1/2), int(h1/2):h1])
    ims.append(dst[int(w1/2):w1, 0:int(h1/2)])
    ims.append(dst[int(w1/2):w1, int(h1/2):h1])

    cv.imshow('1',ims[1])
    cv.waitKey(0)
    cv.destroyAllWindows()
    return ims


