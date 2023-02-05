import cv2
import numpy as np
path1 = r"C:\Users\spval\Downloads\haarcascade_frontalface_default.xml"
path2=r"C:\Users\spval\Downloads\haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier( r"C:\Users\spval\Downloads\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(r"C:\Users\spval\Downloads\haarcascade_eye.xml")
glass=cv2.imread(r"C:\Users\spval\Downloads\cooling glass.jpg")
glassh,glassw,glass_channels=glass.shape
glass_gray = cv2.cvtColor(glass, cv2.COLOR_BGR2GRAY)
_,threshold_img=cv2.threshold(glass_gray, 150, 255, cv2.THRESH_BINARY)
contours,_= cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
trans_img=np.zeros((500,500,3),dtype=np.uint8)
cv2.threshold(trans_img, 150,255,cv2.THRESH_BINARY)
cv2.drawContours(trans_img,contours,-1,(255,255,255),4)
vid_capture=cv2.VideoCapture(0)
while True:
     _,frame=vid_capture.read()
     frame_h,frame_w,frame_channels = frame.shape
     faces=face_cascade.detectMultiScale(frame,1.1,4)
     i=0
     for(x,y,w,h)in faces:
        if i==0:
             i=1
             continue
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h
        glass_width = int( face_w)
        glass_height = int(glass_width * glassh / glassw)
        glass_x1 = face_x2 - int(face_w/2) - int(glass_width/2)
        glass_x2 = glass_x1 + glass_width
        glass_y1 = face_y2 - int(face_h*1.25)
        glass_y2 = glass_y1 + glass_height
        if glass_x1 < 0:
            glass_x1 = 0
        if glass_y1 < 0:
            glass_y1 = 0
        if glass_x2 > frame_w:
            glass_x2 = frame_w
        if glass_y2 > frame_h:
            glass_y2 = frame_h
        glass_width=glass_x2-glass_x1
        glass_height=glass_y2-glass_y1
        glass = cv2.resize(trans_img, (glass_width,glass_height), interpolation = cv2.INTER_AREA)
        roi = frame[glass_y1:glass_y2, glass_x1:glass_x2]
        roi_bg=cv2.bitwise_and(roi,roi)
        roi_fg=cv2.bitwise_and(glass,glass)
        dst=cv2.add(roi_bg,roi_fg)
        frame[glass_y1:glass_y2, glass_x1:glass_x2]=dst
     cv2.imshow('frame',frame)
     cv2.imshow('in',trans_img)
     
     cv2.waitKey(1)
     
     


    
