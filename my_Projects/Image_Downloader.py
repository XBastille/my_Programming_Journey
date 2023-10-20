from PIL import Image
from io import BytesIO     #BytesIO expects bytes like object and produce byte objects. this category of streams can be used for all kinds of non-text data
import cv2 as cv
r=requests.get("https://c8.alamy.com/comp/2RBE3KC/person-taking-picture-by-reflex-camera-man-with-black-photo-camera-in-warm-summer-day-2RBE3KC.jpg")
img=Image.open(BytesIO(r.content))        #BytesIO must be used whenever there is a binary file
img.save("HOLA.jpg")
img2=cv.imread(cv.samples.findFile("HOLA.jpg"))
cv.imshow("HOLA", img2)
if cv.waitKey(0)==ord("k"):
    cv.imwrite("HOLA.jpg")
