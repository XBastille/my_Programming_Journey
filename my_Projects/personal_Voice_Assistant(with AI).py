import speech_recognition as sr
import pyttsx3
import openai
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
from win10toast import ToastNotifier
import os
class initialization:
    def data(self):
        global transai, trans, transweb, transapp, command
        while not command:
            try:
                with mic as source:
                    print("listening...")
                    r.adjust_for_ambient_noise(source, duration=0.5)    
                    r.dynamic_energy_threshold=5000
                    audio=r.listen(source, timeout=5)
                    command=r.recognize_google(audio)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Network error")
        print(command)
        if "stop" in command:
            trans=""
        elif ("who are you" in command 
        or "who build you?" in command 
        or "who have build you" in command 
        or "who created you" in command 
        or "what can you do for me" in command
        or "what is your name" in command
        or "hey cleo" in command 
        or "hey clear" in command):
            trans=command
            command=""
            p=chitchat()
            p.non_AI()
        elif "weather" in command:
            p=other_applications()
            command=""
            p.weather()
        elif (((("launch" in command and ("in chrome" in command or "in web" in command)) or ("open" in command and ("in chrome" in command or "in web" in command))) 
        and ("how to" not in command and "what to" not in command and "where to" not in command)) or 
        (("search" in command or "search for" in command or "look for" in command) and 
        ("how to" not in command and "what to" not in command and "where to" not in command))):
            transweb=command
            command=""
            p=other_applications()
            p.webwork()
        elif (("launch" in command or "open" in command) 
        and ("how to" not in command and "what to" not in command and "where to" not in command)):
            transapp=command
            command=""
            p=other_applications()
            p.system_work()
        elif cleo_check:
            transai=command
            command=""
            p=chitchat()
            p.AI()
    def data_To_Non_ai(self):
        if trans:
            return trans
    def data_To_ai(self):
        if transai:
            return transai
    def data_To_Web(self):
        if transweb:
            return transweb
    def data_To_System(self):
        if transapp:
            return transapp
class chitchat:
    def non_AI(self):
        global cleo_check
        a=initialization()
        datas=a.data_To_Non_ai()
        if "hey cleo" in datas or "hey clear" in datas:
            cleo_check=True
            a.data()
        if cleo_check:
            if "who are you" in datas or "what is your name" in datas:
                self.non_AI_10(
                    'My name is cleopatra, Your personal assistant, you can just call me cleo, I am very pleased to meet you. Just say "hey cleo" and I will answer your call'
                )
                a.data()
            elif "who build you" in datas or "who have build you" in datas or "who created you" in datas:
                self.non_AI_10(
                    "x Bastille has created me. He is a computer science student and he's currently pursuing Artificial intelligence and Machine learning"
                )
            elif "what can you do for me" in datas:
                self.non_AI_10(
                    "here are the things I can do for you:-\n1. I can be your personal AI language model assistant\n2. I can tell you the weather\n3. I can open apps on your desktop but make sure you call by it's exe file name, otherwise the AI won't catch the app that you requested.\n4. I can search, open websites in the web.\nJust say 'hey cleo' and I will answer your call"
                )
            engine.say("So. What can I help you with?")
            engine.runAndWait()
            a.data()     
    def non_AI_10(self, arg0):
        print(arg0)
        engine.say(arg0)
        engine.runAndWait()     
    def AI(self):
        if cleo_check:
            a=initialization()
            datas=a.data_To_ai()
            print("Generating responese...")
            openai.api_key="Your API-Keys here"
            response=openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user", "content":datas}]
            )
            print(response.choices[0].message.content.strip())
            engine.say(response.choices[0].message.content.strip())
            engine.runAndWait()
            engine.say("Is there anything else that I can help you with?")   
            engine.runAndWait()
            a.data()
class other_applications:
    def weather(self):
        if cleo_check:
            a=initialization()
            n=ToastNotifier()
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            r=requests.get("https://weather.com/en-IN/weather/today/l/68509ddcc58030eeb09cd5130641916746139e9174ab3527650e99b108d95ed9", headers=headers)
            soup=BeautifulSoup(r.text, "html.parser")
            current_info1=str(soup.find("div", class_="CurrentConditions--header--kbXKR").get_text())
            current_info2=str(soup.find("span", class_="CurrentConditions--tempValue--MHmYY").get_text())
            current_info3=str(soup.find("div", class_="CurrentConditions--phraseValue--mZC_p").get_text())
            current_info4=str(soup.find("div", class_="CurrentConditions--tempHiLoValue--3T1DG").get_text())
            l=current_info4.split("•")
            res=(f"{current_info1}\n{current_info2}\n{current_info3}\n{current_info4}")
            engine.say("WEATHER UPDATE")
            engine.say(f"{current_info1}\nCurrent temperature is {current_info2}\nIt's {current_info3} today\ntemperature duing day is {l[0]} and during night is {l[1]}")
            engine.runAndWait()
            n.show_toast("WEATHER UPDATE", res, duration=10)
            engine.say("Is there anything else that I can help you with?")   
            engine.runAndWait()
            a.data()
    def webwork(self):
        if not cleo_check:
            return
        a=initialization()
        wish=a.data_To_Web()
        l=wish.split(" ")
        if "open" in wish or "launch" in wish:
            self.webwork_7(l)
        else:
            self.webwork_16(l)
        while True:
            pass
    def webwork_16(self, l):
        wish=" ".join(l[2:])
        engine.say(f"searching for {wish} in web")
        engine.runAndWait()
        wish=wish.replace(" ", "+")
        browser=webdriver.Chrome()
        browser.implicitly_wait(1)
        browser.maximize_window()
        browser.get(f"https://www.google.com/search?q={wish}&start=0")
    def webwork_7(self, l):
        l.pop()
        l.pop()
        wish="".join(l[1:])
        browser=webdriver.Chrome()
        browser.maximize_window()
        browser.get(f"https://www.{wish.lower()}.com/")
        engine.say(f"opening {browser.title}")
        engine.runAndWait()
    def system_work(self):
        if not cleo_check:
            return
        a=initialization()
        wish=a.data_To_System()
        l=wish.split(" ")
        wish="".join(l[1:])
        engine.say(f"opening {wish}")
        engine.runAndWait()
        paths=(r"C:\Users", r"C:\Program Files", r"C:\Program Files (x86)", r"D:", r"E:")
        for path in paths:
            for root, dirs, files in os.walk(path):
                for name in files:
                    if name.lower()==f"{wish.lower()}.exe":
                        os.startfile(os.path.join(root, name))
                        while True:
                            pass
print('say "hey cleo"')
cleo_check=False
command=""
r=sr.Recognizer() 
mic=sr.Microphone()
engine=pyttsx3.init()
voice=engine.getProperty("voices")
rates=engine.getProperty("rate")
engine.setProperty("voice", voice[1].id)
engine.setProperty("rate", rates-50)
al=initialization()
al.data()
