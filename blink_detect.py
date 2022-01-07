#All the imports go here
import numpy as np
import cv2
import time
import os
#Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

#Variable store execution state
first_read = True


#Starting the video capture
cap = cv2.VideoCapture(0)
ret,img = cap.read()
eyes_t=time.time()
face_t=time.time()

while(ret):
    ret,img = cap.read()
    #Coverting the recorded image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray,5,1,1)

    #Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 3,minSize=(200,200))
    
    if(len(faces)>0):
        face_t=time.time()
        
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            #roi_face is face which is input to eye classifier
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,3,minSize=(50,50))

            #Examining the length of eyes object for eyes
            if(len(eyes)>=2):
                eyes_t=time.time()
                #Check if program is running for detection 
                
                cv2.putText(img, "Goz Acik", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
            else:
                eyes_end_time=time.time()
                eyes_two_minutes=eyes_end_time-eyes_t
                cv2.putText(img, f"goz kapali in {int(eyes_two_minutes)} 60 saniye içinde bilgisayar kapatilacak", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
                
                if(eyes_two_minutes>60):
                    os.system("shutdown /s /t 1")
                    
                
               
            
    else:
            face_end_time=time.time()
            face_two_minutes=face_end_time-face_t
            cv2.putText(img, f"Yuz algilanmadi {int(face_two_minutes)} 60 saniye içinde bilgisayar kapatilacak", (70,70), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
            
            if(face_two_minutes>60):
                os.system("shutdown /s /t 1")
        

    #Controlling the algorithm with keys
    cv2.imshow('img',img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break
    

cap.release()
cv2.destroyAllWindows()