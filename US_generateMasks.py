"""
A simple script to generate filled masks from elliptical annotations 
AUTHOR: Ruchi Chauhan
"""
import os
import cv2 as cv
from matplotlib import pyplot as plt

folPath = 'training_set_Labels'
imgList = os.listdir(folPath)

for imgName in imgList:
    img = cv.imread(folPath + '/' + imgName)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, contourIdx=-1, color=(255,255,255),thickness=-1)

    plt.imsave('training_set_masks/' + imgName[:-15] + '_mask.png', img, cmap='gray')