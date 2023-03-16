from flask import Flask
import cv2
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

offset = 20
imgSize = 28

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        
        # img white is where the roi should be indicated, where the shape of the hand is emphasised and remaining extra space is white space
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        
        # outlines where the hand is
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        
        imgCropShape = imgCrop.shape
        aspectRatio = h/w
        
        # aspect ratio segments is to adjust for the hand outline box becoming too large and requires resizing
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            
            imgWhite[:, wGap:wCal+wGap] = imgResize

        elif aspectRatio < 1:
            j = imgSize/w
            hCal = math.ceil(j*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            
            imgWhite[hGap:hCal+hGap, :] = imgResize
            
        cv2.imshow('imagecrop', imgCrop)
        cv2.imshow('imageWhite', imgWhite)
    
    cv2.imshow('image', img)
    
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break