import pyttsx3
#speaking text
'''engine=pyttsx3.init()  #this function gets a reference to a pyttsx3.Engine instance
engine.say("Hello world")
engine.say("How are you doing?")
#engine.save_to_file('Hello World' , 'test.mp3')
engine.runAndWait()
#listening event
def on_start(name):
    print("starting", name)
def on_Word(name, location, length):
    print("word", name, location, length)
def on_end(name, completed):
    print("end", name, completed)
engine=pyttsx3.init() 
engine.connect("started-utterance", on_start)      #connect() registers a callback for notifications of the given topic #"started-utterance" fires an utterance begins
engine.connect("started-word", on_Word)            #fires when it begins speaking a word
engine.connect("finished-utterance", on_end)       #fired when finishes speaking an utterance
engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()
#interupting an utterance
def on_Word(name, location, length):
    print("word", name, location, length)
    if location>0:
        engine.stop()
engine=pyttsx3.init()
engine.connect("started-word", on_Word)
engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()
#changing voices
engine=pyttsx3.init()
voices=engine.getProperty("voices")     #get the current value of an engine property, it features voice, voices, volume, rate 
for i in len(voices):
    engine.setProperty("voice", i.id)    #queues a command to set an engine property. the new property will affect all utterances queued after this function
    engine.say('The quick brown fox jumped over the lazy dog.')
#you can write this loop to get both voices but if you want the female one use engine.setProperty("voice", voices[1].id) without a loop
engine.runAndWait()
#changing rate
engine=pyttsx3.init()
rates=engine.getProperty("rate")
engine.setProperty("rate", rates-100)
engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()
#changing volume
engine = pyttsx3.init()
volume = engine.getProperty('volume')
engine.setProperty('volume', volume-0.75)       #range is (0.0 to 1.0)   note only minus works
engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()'''
def on_start(name):
    print("starting", name)
def on_Word(name, location, length):
    print("word", name, location, length)
def on_end(name, completed):
    print("end", name, completed)
    if name=="fox":
        engine.say("what a  lazy dog", "dog")
    elif name=="dog":
        engine.endLoop()
engine=pyttsx3.init() 
engine.connect("started-utterance", on_start)      #connect() registers a callback for notifications of the given topic #"started-utterance" fires an utterance begins
engine.connect("started-word", on_Word)            #fires when it begins speaking a word
engine.connect("finished-utterance", on_end)       #fired when finishes speaking an utterance
engine.say('The quick brown fox jumped over the lazy dog.', "fox")
engine.startLoop()