import cv2 as cv
import sys
def coordinate(event, x, y, flags, params):
    if event==cv.EVENT_LBUTTONDOWN:
        font=cv.FONT_ITALIC
        cv.putText(img, f"{x},{y}", (x,y), font, 1, (255,0,0), 2)
    elif event==cv.EVENT_RBUTTONDOWN:
        b=img[y,x,0]
        g=img[y,x,1]
        r=img[y,x,2]
        font=cv.FONT_ITALIC
        cv.putText(img, f"{b},{g},{r}", (x,y), font, 1, (0,0,0), 2)
img=cv.imread(cv.samples.findFile("BIBHOR.jpg"))
if img is None:
    sys.exit("Image couldn't be found")
cv.namedWindow("IMAGE")
cv.setMouseCallback("IMAGE", coordinate)
while True:
    cv.imshow("IMAGE", img)
    if cv.waitKey(1)==ord("k"):
        break
cv.destroyAllWindows()
