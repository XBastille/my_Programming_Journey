'''l=[1,2,3,4,5]

l.reverse()
l.insert(3, "insert")
#You can use list function, so that no change can be carried. list has an advantage over [:] as it can be turned into list and u can do mutable changes
l1=list(l)         
l1.append("apollo pharamcy")
print(l1)
print(l)'''
#tuple
'''t="Max", 28, "portoricco", "haha", "Hello"   #tuple actually doesn't require parenthesis
l1,l2, *l3=t                   #packing           # * packs leftover elements of the tuple
print(l1)                      #unpacking
print(l2)
print(l3)'''
'''import sys
l=[0,1,2,"hello",3,4]
t=0,1,2,"hello",3,4
print(sys.getsizeof(l), "bytes")       #a method to know the size of the code
print(sys.getsizeof(t), "bytes")       #it shows that lists take more space than tuple
import timeit
print(timeit.timeit(stmt="[1,2,3,4]", number=1000000))
print(timeit.timeit(stmt="(1,2,3,4)", number=1000000))'''
#dictonary
'''d1={"god":"apollo","human": "iron man", "goddess":"artemis"}
d2=dict(god="shiva", human="spiderman", alien="superman")                # a new way to create a dictionary
print(d2)
d1.update(d2)
print(d1)
a=("nara", "bara")
dict={a:"okay"}                         #we can also add keys like that
print(dict)'''
#set
'''s={12,23,3445,2321, "ulala"}                #as u know sets can only contain unique elements                  
s.add(32)
s.add("plala")                        #add method is used to add elements into a set
s.remove(32)
s.discard(12)      #the main difference between discard and remove is that remove poses an error when the asked to delete is not in the set whereas in discard it doesn't
#clear() clears the set elements
print(s.pop())                        #same as list pop
print(s)
s1={2,4,6,8,10,12}
s2={1,3,5,7,9,11}
s3={2,3,5,7,9,11,13}
print(s2.intersection(s3))
print(s2.union(s3))
print(s2.difference(s3))
print(s2.symmetric_difference(s3))             #symmetric_difference will return all the elements from those two sets but not the elements that are in both sets
s2.update(s3)
print(s1)
s2.intersection_update(s3)             
print(s2)
s2.difference_update(s3)
print(s2)
s2.symmetric_difference_update(s3)
print(s2)
print(s2.issubset(s1))                  #meaning if all the element of s2 are present in s1, it will return true
print(s2.issuperset(s3))                #meaning it will return true if all the element of s3 are present in s2
print(s2.isdisjoint(s3))                #meaning it will return true if those two sets intersection is null
s2=s3.copy()                            #you can copy set using copy() and carry no change if it's modified, we can also use set() 
a=frozenset({1,2,3})                    #frozen set is an immutable version of set, meaning u can do no change such as add, remove, discard etc but u can get the union, intersect etc note(u can't update either)
print(a)'''
#string
'''from timeit import default_timer as timer
#significance of join()
l=["a"]*1000000
#bad python code
start=timer()
s=""
for i in l:           
    s+=i            #string is immutable, in this line, it will create a new string here and then assign it back to our original string, so this opertation is very expensive
stop=timer()
print(stop-start, "sec")

#good code
starts=timer()
n="".join(l)               #using join it makes it much more efficient and faster
stops=timer()
print(stops-starts, "sec")
var=3.453322345
str="the variable is %.10f" % var     #this %s tells that there is a placeholder, %s=string, %d=decimal, %f=float or %.xf=float with x no. of  decimal digits
print(str)
vars=3.453322345
va="cola"
strs="the variable is {:.2f} and {}".format(vars, va)      #:.xf is used to specify the no. of decimal digits
print(strs)
#f-strings are better'''
#collections
'''from collections import Counter                #capital C
var="aaaaabbbbbbbbcccc"        #it can be list, or any any iterable datatype
print(Counter(var))                    #it counts the no. of times the elements present in the datatype occur and return in dictionary, you can also use item(), keys(), values()
#a very big attention to detail,when u only use counter(), Counter({'b': 8, 'a': 5, 'c': 4}) this is the output but when u use .item(), keys, values(), dict_items([('a', 5), ('b', 8), ('c', 4)]) u gat this as the result
#meaning when we use counter u get in ascending order but when u use those dict function u get values according to the varible it is written
print(Counter(var).most_common(10)[0][0])        #if u put more elements than the var as the parameter then u won't get any error, chill!!
#you can get the most common element only, then we can use counter(var).most_common(1)[0][0] or no. of times used only then use [0][1]
print(list(Counter(var).elements()))           #it can turn into list and iterable as well
from collections import namedtuple             #namedtuple is an easy to create light weight object type
Point= namedtuple("dot", "x,y")                #I'm creating a 2D point, what we are doing here is we made a class called dot and the other parameter is all the field I want
pt= Point(2,-5)                                
print(pt)                                      #result is dot(x=2, y=-5)
print(pt.x, pt.y)                              #to access the fields
from collections import   OrderedDict          #old method
hola=OrderedDict()
hola["papa"]=4
hola["nana"]=1
hola["mama"]=2
hola["adaa"]=3
print(hola)
from collections import defaultdict        #similar to assigning a default value in get()
dola=defaultdict(float)          #can be int,list,float in the parameter
dola["papa"]=4
dola["nana"]=1
dola["mama"]=2
dola["adaa"]=3
print(dola["sola"])
from collections import deque     #it's called double queue, u can add or remove from both ends, very efficient
d=deque()
d.append(1)
d.append(2)
d.append(3)                       #u can use appendleft, extendleft, pop(), popleft,clear(),rotate(x)
d.extendleft([5,6,7])             #the result are as follows deque([7, 6, 5, 1, 2, 3]), do u see the uniqueness?
print((d))                       #result are somewhat like this deque([1, 2, 3]), u can change their datatype
d.rotate(-1)                      #before: deque([7, 6, 5, 1, 2, 3]) after: deque([3, 7, 6, 5, 1, 2]) if u want to rotate to the left side use -ve parameter
print(d)'''
#itertools
'''from itertools import product       #it gives the cartestian product of two variable containing elements of any datatype
a=[1,4]
b=[2,4,3]
prod=product(a,b)
rep=product(a,b, repeat=2)         #repeat is kinda wierd lol
print(list(prod))
print(list(rep))'''
'''from itertools import permutations
a=[1,2,3,4]
perm=permutations(a, 2)       #you can also put length to your permutations as ur second arguement in order to have shorter permuatations
print(list(perm))
from itertools import combinations, combinations_with_replacement
a={1,2,3,4}
com=combinations(a,3)
comr=combinations_with_replacement(a,2)
print(set(com))
print(set(comr))
from itertools import accumulate       #it helps to return accumulated sums
import operator            #so accumulate module has set giving sums as default, id we want something else then we can import operator
a=[1,2,3,4]
b=[1,2,5,4,3,7,6,8]
acc=accumulate(a) 
accc=accumulate(a, func=operator.mul)     #here it's is multiplied
man=accumulate(b, func=max)     #this will give the max for each comparison       
print(list(acc))               #result we got is this [1, 3, 6, 10]
print(list(man))               #result we got is this [1, 2, 5, 5, 5, 7, 7, 8]
print(list(accc))'''
'''from itertools import groupby
a=[1,2,3,4]
def smaller_than_3(x):
    return x<3
gr=groupby(a, key=smaller_than_3)
gg=groupby(a, key=lambda x : x<3)  #making a  user function is not required here as we used the lambda function
persons=[{'name':"hola", "age":"25"}, {'name':"sola", "age":"25"}, {'name':"hyperbola", "age":"27"}, {'name':"colahola", "age":"23"}]
gh=groupby(persons, key=lambda x: x["age"])
for key, value in gr:
    print(key, list(value))        #True [1, 2], False [3, 4] are the result, what groupby does is it groups the element and put it in lists in depending on the condition we have put here, the codition is less than 3.
for key, value in gg:
    print(key, list(value))
for key, value in gh:             # 5 [{'name': 'hola', 'age': '25'}, {'name': 'sola', 'age': '25'}] is the result and u can see it groupby age very cool and amazing
    print(key, list(value))       #27 [{'name': 'hyperbola', 'age': '27'}]
                                  #23 [{'name': 'colahola', 'age': '23'}]
from itertools import count, cycle, repeat   #infinite iterators
for i in count(10):           #count function will start counting from x it won't end until u break it
    print(i)
    if i==15:
        break
a=[1,2,3]
for i in cycle(a):          #it will cycle infinitely through an iterable
    print(i)
    break
for i in repeat(1, 4):
    print(i)'''
#lambda
'''#lambda function is tipically used when u need a simple function that is used only once in your code
#or it is used as a arguement to higher order functions meaning the functions that take in other functions as arguements
points2D=[(2,3), (1,3), (-10,4), (3,5), (6,7)]
points2D_sorted=sorted(points2D, reverse=True, key=lambda x:x[1])    #a key usually requires a function 
points2D_sorted2=sorted(points2D, reverse=True, key=lambda x:x[1]+x[0])   #here we used a different method to sort, we can sort in any ways we want, we just have to use our creativity to solve question
print(points2D)
print(points2D_sorted)         # by default it gives [(-10, 4), (1, 3), (2, 3), (3, 5), (6, 7)] this meaning it sorts based on 1st arguement only'''
 #error

'''a= 5+'10'    #typeerror
import somemodule         #ModuleNotFoundError   
a=5
b=c                       #NameError
f=open("filenono")     #FileNotFoundError
a=[1,2,3]         
a.remove(4)               #ValueError(it happens if the function or a operation recieves an arguement that has a right type but an inappropiate value)
a[4]                            #IndexError
d={"name":"Bibhor"}
d["age"]                   #KeyError
x=-5
assert (x>=0),"x is not positive"    #you can also use "raise"
#you can define your own exception by simply defining our own error class
class ValueTooHighError(Exception):         #capital E
    pass 
class ValueTooSmallError(Exception):
    def __init__(self, message, value):
        self.message=message
        self.value=value

def hola(s):
    if s>100:
        raise ValueTooHighError("value is too high!")      #the result we got by simply calling hola(4) is _main__.ValueTooHighError: value is too high!
    if s<7:
        raise ValueTooSmallError("value is too low!!", s)
try:
    hola(1)
except ValueTooHighError as e:
    print(e)
except ValueTooSmallError as e:
    print(e.message, e.value)'''
#logging
'''import logging
#https://docs.python.org/3/library/logging.html
#https://docs.python.org/3/library/time.html
#log levels
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d/%m/%Y %H:%M:%S")
import helper
logging.debug("This is gonna debug")
logging.info("we need info bois!!")
logging.warning("Imma give u a warning boi")           #only warning,error,critical will be printed if basicconfig is not used because by default python print warning level and above
logging.error("You're doing errors boi")
logging.critical("boi, we have reached a critical error!!")'''
#will be continue
#json-javascriptobjectnotation, it's a lightweight data format that is used to exchange data and is heavily used in web application
#from python to json(serialization, encoding)
'''import json
person = {"name": "John", "age": 30, "city": "New York", "hasChildren": False, "titles": ["engineer", "programmer"]}
print(json.dumps(person ,indent=4, separators=(";", "=")))           #indent=4 is recommended  #it's not recommended to use different separators, use the default ones
personJSON=json.dumps(person ,indent=4, sort_keys=True)
print(personJSON)      #arranged the keys in ascending order of course
#we can also dump or convert into file
with open(person.json, "w") as file:
    json.dump(person, file, indent=4)      #DOUBT
#from json to python (deserialization,decoding)
person=json.loads(personJSON)
print(person)
#we can also decode a json file
with open("intermediate_note.json", "r") as file:
    hola=json.load(file)
    print(hola)
#this is how a json looks like filename=intermediate_notes.json (serlialization done above)
{
    "firstName": "Jane",
    "lastName": "Doe",
    "hobbies": ["running", "swimming", "singing"],
    "age": 28,
    "children": [
        {
            "firstName": "Alex",
            "age": 5
        },
        {
            "firstName": "Bob",
            "age": 7
        }
    ]
}
#note:- the dumps meaning dump from a string whereas dump meaning dump from a json same goes for load
#if we want to create a custom object
class User:
    def __init__(self, name, age):
        self.name=name
        self.age=age
user=User("Bibhor", 18)
def encode_user(o):      #this is a custom encoding function
    if isinstance(o, User):                   #it checks whether an object is an instance of a class
        return {"name":o.name, "age":o.age,o.__class__.__name__:True}       #this is a little trick which will get the class the name as a key
    else:
        raise TypeError ("Object of type User is not JSON serializable")
#alternate method
from json import JSONEncoder
class Userencoder(JSONEncoder):
    def default(self,o):
        if isinstance(o, User):                   
            return {"name":o.name, "age":o.age,o.__class__.__name__:True}
        return JSONEncoder.default(self,o)                  #otherwise we let the base JSONEncoder handle it
userJSON=json.dumps(user, default=encode_user, indent=4)
usersJSON=json.dumps(user, cls=Userencoder, indent=4)
#other way of calling
userJson=Userencoder(indent=4).encode(user)
print(userJSON)                         #TypeError: Object of type User is not JSON serializable
print(usersJSON)
print(userJson)

def decode_user(dct):
    if User.__name__ in dct:                       #we were able to use this because we had included the class name in the dict
        return User(dct["name"], dct["age"])
    return dct
hola=json.loads(userJson, object_hook=decode_user)
print(hola.name)'''
#random

'''import random
print(random.random())          #it generate a float between 0 to 1
print(random.uniform(0,10))     #same as above one but we can put range
print(random.randint(0,10))
print(random.randrange(0,10))   #note:-the difference between int and range is that 10 as a result is possible in int whereas in range it stays within the range
#https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_PDF.svg
print(random.normalvariate(0,1))   #(mu,sigma) mu=mean, sigma= standard deviation
#random.choice
a=list("ABCDEFGHIJK")
random.seed(1)                     #because it is reproduceable it is not recommended for security purposes
print(random.sample(a, 3))         #same as choice but it returns more unique elements as per your choice it returns in original datatype 
print(random.choices(a, k=3))      #same as sample but not unique
#random.shuffle()
import secrets                    #it is used for passwords, security tokens or account authentication etc
#disadvantage it takes more time in these algorithims but it will generate a true random no.
print(secrets.randbelow(10))
print(secrets.randbits(4))      #bits means  binary digits, so for eg 4 bits means 4 binary digits for eg 1010 so the highest no. will be 1111 i.e 15 so, it will give from 0 to 15
l=list("ABCDEFG")
print(secrets.choice(l))
#basically secrets is like random but it is not reproducable
import numpy as np
np.random.seed(1)
print(np.random.rand(3,3))        #the arguement is diemension, range is 0 to 1, gives float, can also be (3)
print(np.random.randint(0,10,(3,3)))  #we give the range (left) and diemension (right) as arguement
a=np.array([[1,2,3], [4,5,6], [7,8,9]])
print(a)
np.random.shuffle(a)
print(a)'''
#decorator         #there are two types of decorators:- function decorator, class decorator more common is the function decorator
                   #a decorator is a function that takes another function as an arguement and extends the behaviour of this function without explicitly modifying it in another words it adds new functionality to the existing function
'''import functools
def start_end_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):         #with this syntax you can use as many arguements and keyword arguements as possible
        print("before")                   #so, this is a nice template for a decorator 
        res=func(*args, **kwargs)
        print("after")
        return res
    return wrapper
@start_end_decorator      #in other word v=start_end_decorator(add5)
def add5(x):
    print(x+5)
add5(10)
print(help(add5))
print(add5.__name__)              #__name__ meaning the name of the functions, here we are trying to print the name of the functions 
                                    #Help on function wrapper in module __main__:
                                    #wrapper(*args, **kwargs)
                                    #None
                                    #wrapper
                                    #this is the result we got which is wrong, python actually got confused but we can fix this by importing functools

def repeat(num_times):                     #we have to make two functions, one for to take the arguement given by decorator
    def decorator(func):                   #another for taking the function(here the func is greet) 
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                res=func(*args, **kwargs)
            return res
        return wrapper
    return decorator
@repeat(6)                            #we can also pass arguements thru decorator
def greet(name):
    print(f"Hello {name}")
greet("Bibhor")
#nested decorators
def start_end_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):         
        print("start")                   
        res=func(*args, **kwargs)
        print("end")
        return res
    return wrapper
def debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)               #doubt
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {result!r}")
        return result
    return wrapper
@debug
@start_end_decorator
def say_hello(name):
    greeting=f"hello {name}"
    print(greeting)
    return greeting
print(say_hello("Bibhor"))
#class decorators
class CountCalls:
    def __init__(self, func):
        self.func=func
        self.num_calls=0
    def __call__(self, *args, **kwargs):      #__call__ is awesome, it enables to execute object just like a function, here we used cc() to call function inside a class just like a function calling
        self.num_calls+=1
        print(f"the function is executed {self.num_calls} times")
        return self.func(*args, **kwargs)
@CountCalls
def say_hello(name):
    print(f"hello {name}")
for _ in range(100):
    say_hello("Bibhor")
#Some typical use cases
#Use a timer decorator to calculate the execution time of a function
#Use a debug decorator to print out some more information about the called function and its arguments
#Use a check decorator to check if the arguments fulfill some requirements and adapt the bevaviour accordingly
#Register functions (plugins)
#Slow down code with time.sleep() to check network behaviour
#Cache the return values for memoization (https://en.wikipedia.org/wiki/Memoization)
#Add information or update a state'''
#generator
#Generators are functions that can be paused and resumed on the fly, returning an object that can be iterated over.
#they generate only items one at a time and only when u asked for it, because of that they're much more memory efficient than other sequence objects when you have to deal with large data sets
#they are powerful and advanced python technique
'''def generate():
    yield 2
    yield 1
    yield 3
g=generate()
for i in g:
    print(i)
#You can also get value 1 at a time
print(next(g))
print(next(g))    #basically next() pauses at yield and waits to get started when another next() is used
print(next(g))
#if you write next after this u will get an error as there is no value left to generate cause we have yielded 3 values only
print(sum(g))    #return the sums of the values we have yielded
print(sorted(g))    #this will ofc return in list sorted
#execution of generator in detail
def countdown(num):
    print("starting")              #very important note, if u run without next() then python will nothing as this function is a generator and generators won't return unless u order them to 
    while num>0:    
        yield num                  #calling using next, it remembers the value the num possess currently
        num-=1
cd=countdown(4)
value=next(cd)
print(value)
import sys
def firstn(n):
    l=[]
    num=0
    while num<n:
        l.append(num)        #in this way, all the no. stored in the list, so, it takes a lot of memory
        num+=1
    return l
def firstn_generator(n):
    num=0
    while num<n:
        yield num                   #one thing we use generator is when we are required to make a list of huge no. of elements
        num+=1
print(sys.getsizeof(firstn(1000000)),"bytes")
print(sys.getsizeof(firstn_generator(1000000)), "bytes")
#another advantage is we don't have to wait until all the elements have been generated before we start to use them for example we can get the first element by using 1st next() and we don't have to calculate all the elements
def fibonacci(limit):
    a,b=0,1
    while a<limit:
        yield a
        a,b=b, a+b
fib=fibonacci(100)
for i in fib:
    print(i, end=", ")
#generators expression (basically it is like list comprehension)
mygenerator=(i for i in range(10) if i%2==0)
for i in mygenerator:                  #again it takes much less space than list comprehension
    print(i)'''
#function arguments
#Variable-length arguments (*args and **kwargs)         #args can take up any no. of positional arguements to your functions, you are generally called args and kwargs but we can call typically anything we want
#kwargs can take up any no. of keyword arguements
#args are tuples, whereas kwargs are dictionary
'''def foo(a, b, *args, **kwargs):
    print(a,b)
    for arg in args:
        print(arg)
    for key in kwargs:
        print(key, kwargs[key])
foo(1,2,3,4,5, six=6, seven=7)
#def foo(a, b, *, c, d):     #after star only keyword arguement is required, else error
#def foo(*args, c, d):     #same here #these are called forced parameters
#Container unpacking into function arguments
def fool(a,b,c):         #important note:- if value to other parameters is not given then error won't occur, python will simply ignore
    print(a,b,c)
    print(c+5)
my_list=[1, 2, 3]       #it can also be tuple
fool(*my_list)  
my_dict={"a":1, "b":2,"c":3}
fool(**my_dict)'''
#asterisk
#used for multiplication and power operation, unpacking to another list or as an argument, use as *args, **kwargs
'''t=(1,2,3)
l=[4,5,6]
new_list=[*t, *l]
print(new_list)
d1={"a":1, "b":2, "c":3}
d2={"c":5, "d":6}
d3={**d1,**d2}
print(d3)'''
#shallow copy vs deep copy
#-shallow copy:one level deep, only reference of nested child objects
#-deep copy:full independent copy
#shallow copy
'''import copy
org=[0,1,2,3,4]
cpy=copy.copy(org)   #org.copy() and list() and org[:] are also under shallow copy 
cpy[0]=-10
print(cpy)
print(org)
org=[[0,1,2,3,4], [5,6,7,8]]
cpy=copy.deepcopy(org)
cpy[0][1]=-10
print(org)
print(cpy)
class Person:
    def __init__(self, name, age):
        self.name=name
        self.age=age
class Company:
    def __init__(self, boss, employee):
        self.boss=boss
        self.employee=employee
pe=Person("Bibhor", 55)
ps=Person("Aditya", 27)
pt=copy.copy(pe)
pt.age=57
print(pt.age, pe.age)
company=Company(pe, ps)       #like we said we can do this, although we can't print it like that but we can transfer object so, no error
company_clone=copy.deepcopy(company)      #we had to do deepcopy here as it has nested instance i.e inside Company's instance we have Person's instance
company_clone.boss.age=56
print(company_clone.boss.age, company.boss.age)
'''
#context managers
#context managers are a great tool for resource management, they allow us to allocate and release resources precisely when we want to
#context manager for our own custom classes by implementing a class with __enter__ and __exit__
'''class ManagedFile:
    def __init__(self, filename):
        print("__init__")
        self.filename=filename
    def __enter__(self):          #as soon as the with statement is called, the enter gets called
        print("enter")
        self.file=open(self.filename, "w")
        return self.file
    def __exit__(self, exc_type, exc_value,exc_traceback):          #here we make sure that we correctly close the file
        if self.file:                        #if it is not none
            self.file.close()
        if exc_type is not None:
            print("Exception has been handled")      #if we want to handle this exception on our own, then we must return True. Note:-but the contents of the file won't be printed
        #print("exc:", exc_type, exc_value)
        print("exit")
        return True
with ManagedFile("dry_run.txt") as file:
    print("to do stuff....")
    file.write("to do stuffs")
    file.somemethod()
print("continuing")'''
#you can also implement as a function
from contextlib import contextmanager
@contextmanager
def open_managed_file(filename):
    f=open(filename, "w")
    try:                        #here it does the work as __enter__
        yield f    
    finally:
        f.close()               #here it does the work as __exit__
with open_managed_file("dry_runlol.txt") as file:
    file.write("lolol")
