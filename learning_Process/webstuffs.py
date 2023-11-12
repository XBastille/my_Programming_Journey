import requests
'''r=requests.get("https://api.github.com/events")     #we have a reponse object called r, we can get all the info we need from this object
r=requests.post("https://httpbin.org/post", data={"key":"value"})   #HTTP post request
r=requests.put("https://httpbin.org/put", data={"key":"value"})
r=requests.delete("https://httpbin.org/delete")
r=requests.head("https://httpbin.org/put")
r=requests.options("https://httpbin.org/put")
#What is a requests?
#when a  client asks for a resource to a server, that is called a request
#What is a response?
# when the server provides the resources  that it was requested by the client is called response   
#what is get?
the get method means retrieve whatever information  is identified by the Request-URI(URI identifies the resource to which requests applies). IF the Request-URI refers to a data-producing process, it is the produced data which shall be returned as the entity in the response and not the source text of the process, unless that text happens to be the output of the process
#What is POSt?
#Post method submits data to be processed to the identified resource
#the main difference between GET and POST is get carries request parameter in the URL whereas POST takes carries request parameter in message body which makes it more secure than get, and it's the reason why we use POST to share password and other private information
#we often want to send data in the URL's query list, we send data in key/value pairs in the URL after the question mark, requests allows these arguements in the form of dictionary of strings
#EG:-httpbin.org/get?key=val
payload={"key1":"value1", "key2":"value2", "key3":["value3", "value4"]}
r=requests.get("https://httpbin.org/get", data={"X_Bastille":"099012"}, params=payload)      #httpbin.org is a great tool to test requests
print(r.json())           #we can do the json serialization if we want by importing json
print(r.url)
print(r.text)
payload={"key1":"value1", "key2":"value2", "key3":["value3", "value4"]}
r=requests.post("https://httpbin.org/post", data={"key":"value"}, params=payload)
print(r.text)           #requests will automatically decode contents from the servers
print(r.encoding)       #the text encoding which is guessed by requests is used when we access r.text
r.encoding='ISO-8859-1'
print(r.encoding)
#we might want to change the encoder like above when there is a special logic required to know what the encoding of the content will be
#for eg:-  html and xml have the ability to specify their encoder in their body, for that we have to use r.content to know the encoder then change using r.encoding and then use r.text for proper encoding
#we can create our own custom encoding
print(r.content)       #it gives the response body as bytes
print(r.json())        #note:-success of the  call to r.json()  doesn't indicate the success of response.
print(r.raise_for_status())
print(r.text)
q=requests.get("https://www.youtube.com")
with open("index.html", "w", encoding="utf-8") as file:
    file.write(q.text)
r=requests.put("https://httpbin.org/put", data={"key":"value"})
print(r.text)
#IMP:-put requests is used when we want to create something
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
#what is a header?
#header method retrives more details about the information we want to get
#you can download other stuffs as well
#r=requests.get("https://www.win-rar.com/fileadmin/winrar-versions/winrar/winrar-x64-624.exe")  #file download'''
from tqdm import tqdm
r=requests.get("https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_win32.zip", stream=True) 
totalExpectedBytes=int(r.headers["content-length"])
'''with open("goofy_ahh_sound2.wav", 'wb') as fd:
    for chunk in r.iter_content(chunk_size=128):          #here we are basically setting up the download speed, i.e chuck size bytes per sec 
        print(f"{bytesReceived} Bytes out of {totalExpectedBytes} Bytes")
        fd.write(chunk)            #download files don't require bytesIO as they are not binary, they are simply download links
        bytesReceived+=1000000'''
progress_Bar=tqdm(total=totalExpectedBytes, unit="B", unit_scale=True, colour="green")
with open("chromedriver.zip", 'wb') as fd:
    for chunk in r.iter_content(chunk_size=1024):
        progress_Bar.update(len(chunk))
        fd.write(chunk)
progress_Bar.close()
