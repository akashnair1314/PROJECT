import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

folder="Assets/K"
counter=0
cap=cv2.VideoCapture(0)#0 is the id for the web cam
# running an infinite loop
offset=20#offset variable is created to add buffer space in the cropped image
imagesize=300#this will be the size of all the photos for training
detector = HandDetector(maxHands=1)
while True:
    success,img=cap.read()#success is a boolean variable which stores the status of cap object,img contains the frame
    hands,img=detector.findHands(img)
    #creating a bounding box if a hand is detected
    if hands:
        hand=hands[0]
        #setting the values for the box
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((imagesize,imagesize,3),np.uint8)*255#created a 2d array which is going to store values btw 0-255 i.e unit8 (unsigned interger of 8 bits multiplyi8ng whith 255 makes all values to 255 making the pic white)
        #storing it in a new image
        imgcrop=img[y-offset:y+h+offset,x-offset:x+w+offset]#starging height is y and endign height is y+h similary for the width (x)
        imgcropshape=imgcrop.shape
        aspectratio=h/w
        #math.ceil is used for taking the upper bound value to make the values similar
        #aspect ratio is taken to find which is greter and according to that lenght or the width will be stretched
        #if height is greater than width
        if aspectratio>1:
            k = imagesize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop, (wCal, imagesize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imagesize - wCal) / 2)
            imgwhite[:, wGap:wCal + wGap] = imgResize
        else :
            k = imagesize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imagesize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imagesize - hCal) / 2)
            imgwhite[hGap:hCal + hGap, :] = imgResize
        #offset added to keep buffer space
        cv2.imshow("Imagecrop",imgcrop)
        cv2.imshow("Imagewhite",imgwhite)

    cv2.imshow("Image",img)
    key= cv2.waitKey(1)
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)