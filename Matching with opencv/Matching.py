# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:25:19 2021

@author: faruk
"""

import os
import cv2
import numpy as np

path =r"C:\Users\faruk\OneDrive\Masaüstü\classification\val" # class path
images=[]
classnames=[]
pathlist=os.listdir(path)
for cl in pathlist:
    imgx=cv2.imread(f'{path}/{cl}',0)
    images.append(imgx)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


sift = cv2.xfeatures2d.SIFT_create()


def findingDes(images):
    desList=[]
    for img in images:
        kp,des= sift.detectAndCompute(img, None)
        desList.append(des)
    return desList

desList=findingDes(images)


def findingid(image,desList,treshold=40):
    kp,des1= sift.detectAndCompute(image, None)
    bf = cv2.BFMatcher()
    machlist=[]
    finalvalue=-1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des1, k = 2)
            goodmatch = []
            for match1, match2 in matches:
                
                if match1.distance < 0.75*match2.distance:
                    goodmatch.append([match1])
            machlist.append(len(goodmatch))
    except:
        pass
    if len(machlist)!=0:
        if max(machlist)>treshold:
            finalvalue =machlist.index(max(machlist))
    return finalvalue
cap = cv2.VideoCapture(0)

while True:

    _, targetimage = cap.read()
    imgshow = targetimage.copy()
    targetimage=cv2.cvtColor(targetimage,cv2.COLOR_BGR2GRAY)
    
    
    
    id1=findingid(targetimage, desList)
    if id1!=-1:
        cv2.putText(imgshow, classnames[id1], (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    
    cv2.imshow("0", imgshow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):break


cap.release()
cv2.destroyAllWindows() 
    




