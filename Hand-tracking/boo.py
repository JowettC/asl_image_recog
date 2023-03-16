from flask import Flask
import cv2
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import tensorflow as tf

cap = cv2.VideoCapture(0)
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
while True:
        ret,img=cap.read()  #Read from source
        height, width, _ = img.shape
        
        roi= img[100:320, 100:320]
        
        
        mask = object_detector.apply(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > 100:
                cv2.drawContours(ret, [cnt], -1, (0,255,0), 3)
        
        
        if(cv2.waitKey(10) & 0xFF == ord('b')):
          break #break when b is pressed 
        cv2.imshow('roi', roi)
        cv2.imshow('frame', img)
        cv2.imshow('mask', mask)