 import cv2 as cv
import numpy as np
import sys
import imutils
'''flags = [i for i in dir(cv) if i.startswith("COLOR_")]
print(flags)
img=cv.imread(cv.samples.findFile("BIBHOR.jpg"))   #gets the file  
if img is None:
    sys.exit("image couldn't be found!")           #checks whether the file is valid or not
cv.imshow("Display window", img)                  #shows the file
k = cv.waitKey(0)         #here, zero means to wait forever
if k==ord("s"):
    cv.imwrite("BIBHOR.jpg", img)              #the image is written to a  file if "s" is typed'''
cap=cv.VideoCapture(0)       #here the arguement can be a device index or name of the video file, here device index means what camera channel you want to use, simply give 0(-1) to play the primary camera, or secondary camera to use, type 1
if not cap.isOpened():
    sys.exit("Failed to open camera!")
while cap.isOpened():
    '''capture frame by frame'''
    ret, frame=cap.read()       #here ret will get either true or false
    '''if frame is read correctly, then ret is true '''
    if not ret:
        print("error, frame is not receiving correctly, exiting....")
        break
    '''Our operations on the frame come here'''
    #frame=imutils.rotate(frame, angle=0)
    frame=imutils.resize(frame, width=768)
    colour=cv.cvtColor(frame, cv.COLOR_BGR2HLS_FULL)  #we can put any color we want
    cv.imshow("frame", colour)
    if cv.waitKey(1)==ord("q"):
        break
    '''when everything done, release the capture'''
cap.release()             #Videowriter has to be released as well
cv.destroyAllWindows()
