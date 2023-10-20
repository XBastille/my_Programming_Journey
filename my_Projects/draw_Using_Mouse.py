import cv2 as cv
import numpy as np
drawing1=False  #True if mouse is pressed
drawing2=False #for rectangle
ix,iy=-1,-1
'''mouse callback function'''
def nothing(x):
    pass
def draw_circle(event,x,y,flags,param):             #x,y is the coordinate of the every mouse event
    global drawing1, drawing2, ix, iy, r,g,b,ra
    if s==1:
        cv.rectangle(img, (0, 0), (1024, 512), (b, g, r), -1)
    if event==cv.EVENT_LBUTTONDOWN:
        drawing1=True
        ix,iy=x,y
    elif event==cv.EVENT_MOUSEMOVE:
        if drawing1:
            cv.circle(img, (x,y), ra, (b,g,r), -1)         #here event is mouse event meaning all the functions the mouse does, such as left bottom up, left bottom down etc
        '''elif drawing2:
            cv.rectangle(img, (ix,iy), (x,y), (255,0,0), -1)'''
    elif event==cv.EVENT_LBUTTONUP:
        drawing1=False
    '''if event==cv.EVENT_RBUTTONDOWN:
        drawing2=True
        ix,iy=x,y'''             #not in correct position but who cares
    '''elif event==cv.EVENT_RBUTTONUP:
        drawing2=False'''
'''Create a black image, a window and bind the function to window'''
img=np.zeros((512, 1024, 3), np.uint8)
cv.namedWindow("IMAGE")
cv.createTrackbar("R", "IMAGE", 0, 255, nothing)                  #it assigns a variable to be a positional synchronized with the trackbar and specifies a callback functions onchange to be called on the trackbar position change
cv.createTrackbar("G", "IMAGE", 0, 255, nothing)
cv.createTrackbar("B", "IMAGE",0,255,nothing)
cv.createTrackbar("radius", "IMAGE", 0, 100, nothing)
switch="0 : OFF /n1 : ON"
cv.createTrackbar(switch, "IMAGE", 0, 1, nothing)
cv.setMouseCallback("IMAGE", draw_circle)             #this function takes place whenever there is a mouse event (namedwindow, mouse callback function)
while True:
    cv.imshow("IMAGE", img)
    if cv.waitKey(1)==ord("q"):
        break
    r=cv.getTrackbarPos("R", "IMAGE")              #returns the current trackbar position
    g=cv.getTrackbarPos("G", "IMAGE")
    b=cv.getTrackbarPos("B", "IMAGE")
    ra=cv.getTrackbarPos("radius", "IMAGE")
    s=cv.getTrackbarPos(switch, "IMAGE")
cv.destroyAllWindows()
