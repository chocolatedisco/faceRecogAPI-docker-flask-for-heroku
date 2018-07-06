# -*- coding: utf-8 -*-
import os
import cv2
import tensorflow
import numpy as np
import keras
from os import path
from flask import Flask
from keras.models import load_model

img = cv2.imread("./asuka.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
face_list=cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
#顔が１つ以上検出された時
if len(face_list) == 1:
    for rect in face_list:
        x,y,width,height=rect
        img = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        if img.shape[0]<64:
            continue
        img = cv2.resize(img,(64,64))
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        img = np.array(img)
        model = load_model("./FaceRecog.h5")
        ans = model.predict(img)
print(ans)
