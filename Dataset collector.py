# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 01:11:12 2022

@author: Ahmed Amr
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

folder = "Dataset/5"
offset = 20 # space for crop hand
imgSize = 300 # to ensure size is one for all
cap = cv2.VideoCapture(0) #open camera
detector = HandDetector(maxHands=1) 
counter = 0
while True:
    success , img = cap.read() #read the frame
    hands = detector.findHands(img,draw=False) #detect hands from img
    if hands:
        hand = hands[0] #because there is one hand 
        x,y,w,h = hand['bbox'] #bounded box its four points to crob hand from img
        
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #second frame for edit the main frame 
        
        imgCrop = img[y-offset : y+h+offset , x-offset : x+w+offset]
        
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w  # for check to make the img 300*300 max if h>w or w>h in white img
        # for handling the over hight
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[: , wGap:wCal+wGap] = imgResize
            
        # for handling the over width    
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap , :] = imgResize
        
        cv2.imshow("image Crop", imgCrop)
        cv2.imshow("image White", imgWhite)
        
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
        
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()  