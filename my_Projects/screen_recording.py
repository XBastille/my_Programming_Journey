import cv2 as cv
import numpy as np
from pyautogui import screenshot    #it lets your python scripts control the mouse and keyboard to automate interactions with other application
fourcc=cv.VideoWriter_fourcc(*"DIVX")
out=cv.VideoWriter("output.avi", fourcc, 60.0, (1920, 1080))         #(filename, video codec, fps, resolution)
cv.namedWindow("screen_recording", cv.WINDOW_NORMAL)
cv.resizeWindow("screen_recording", 480, 270)
while True:
    img=screenshot()
    frame=np.array(img)
    frame=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    out.write(frame)
    cv.imshow("screen_recording", frame)
    if cv.waitKey(1)==ord("k"):
        break
out.release()
cv.destroyAllWindows()
