import pyscreenshot as ImageGrab
import cv2 as cv
from time import sleep
import imutils
from win10toast import ToastNotifier
n=ToastNotifier()
sleep(3)
img=ImageGrab.grab()
img.save("Q10_1092.jpg")
img2=cv.imread(cv.samples.findFile("Q10_1092.jpg"))
#img2=imutils.resize(img2, width=1024)
cv.imshow("IMAGE", img2)
if cv.waitKey(0)==ord("k"):
    cv.imwrite("Q10_1092.jpg", img2)
n.show_toast("Screenshot\nScreenshot saved to images folder")
