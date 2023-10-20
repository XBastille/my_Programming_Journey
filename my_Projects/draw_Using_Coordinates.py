import cv2 as cv
import numpy as np
while True:
    '''create a black image'''
    img=np.zeros((512,1024, 3), np.uint8)        #seems like (512, 512, 3) represents black
    #we also have personal img or video as the background
    cv.rectangle(img, (0, 0), (1024, 512), (0, 0, 255), -1)       #here we have coordinates of top left corner and bottom right corner of the rectangle
    '''draw a digonal blue line of thickeness 0f 5 px'''
    cv.line(img, (0, 0), (2048,1024), (255,255,255), 5)          #(img, cordinates of the diagonal line, the color, thickness)
    cv.rectangle(img, (562, 256), (828, 128), (255, 0, 0), 5)
    cv.circle(img, (684, 192), 60, (0, 255, 0), 5)
    #we can write -1 as thickness meaning filling inside the shape(for closed objects only)
    font=cv.FONT_ITALIC
    cv.putText(img, "X_BASTILLE", (10,500), font, 4, (255,255,0), 5, cv.LINE_AA)    #(img, text, coordinate, font, font size, color, thickness, cv.LINE_AA)
    cv.imshow("image", img)                                                         #note:-there's no visible difference in using cv.LINE_AA
    if cv.waitKey(1)==ord("q"):
        break
