import speech_recognition as sr
r=sr.Recognizer()      #recognize instance is to of course recognize speech
mic=sr.Microphone()     
try:
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.5)    #default is 1. it is recommended to set it to 0.5 because by default this functions reads the first second of the audio and calibrates the recognizers to the noise level of that audio
        r.dynamic_energy_threshold=5000
        audio=r.listen(source, timeout=5)
except sr.WaitTimeoutError:
    pass
except sr.UnknownValueError:
    pass
except sr.RequestError:
    print("Network error")
print(r.recognize_google(audio))    #THERE are many different types of recognize(). they require key/token
'''harvard=sr.AudioFile("harvard.wav")
with harvard as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
    audio1=r.record(source, duration=4)          #record() records the data in the audio file   
    audio2=r.record(source, duration=4)      
print(r.recognize_google(audio1, show_all=True))
print(r.recognize_google(audio2))'''
