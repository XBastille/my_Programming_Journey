 '''a=True

b=[]
c=[1,2,3,4,5,6]
print(all([a,b,c]))'''
'''a=complex(2,5)
b=5
print(b.real,a.imag)'''
'''c=8
print(c.bit_length())    # this functions returns the of bits required to represent the no. in binary
'''
'''print(abs(-7.2))
print(round(6.768,2))'''
'''l=[12,34,56,78,91]
l[1:1]+=[12,34,6546,2334,667]   #another cool way to add multiple elements
print(l)'''
'''l=["car","bike","nice", "Bzr", "fat", "Can"]
l1=l[:]
l.sort(key=str.lower)   # very cool stuff
print(l)
print(l1)'''
'''l=["car","bike","nice", "Bzr", "fat", "Can"]
print(sorted(l, key=str.lower))
print(l)'''
'''l=["car","bike","nice", "Bzr", "fat", "Can"]
print(l.index("bike"))        # also applicable for tuple
'''
'''a=("car","bike","nice", "Bzr", "fat", "Can")
print(sorted(a,  key=str.lower))     # result will be in list as tuple can't be modified
b=a + ("jar", "far")           #  a way to add elements in tuple but not in original tuple
print(b)'''
'''d={"car":"Tata", "bike":"Honda", "cycle":"Goana", "truck": "Mercedes"}
print(d.get("airplane", "Air India"))       # using get function you can use a  default value if the key is  not found
d.pop("car")
d.popitem()
d.[fruit]="mango"                           # add keys along with values
print(d)
print(tuple(d.items()))
d1=d.copy()           #using copy, the duplicate variable doesn't carry the values or use dict()
print(d1)'''
'''#sets
s1={"nice", "solo"}
s2={"solo"}
union=s1 or s2  # we can use |
intersect=s1 and s2   # we can also use &
print(intersect)
print(union)
diff= s1-s2  # diff. between two set
print(diff)
subset=s1>s2
print(subset)'''
'''def count():
    count=0
    def increment():
        nonlocal count      # use variable outside the inner function but inside outer function
        count+=1
        print(count)
    increment()
    increment()
count()'''
'''item=['apple', "Banana", "orange"]
for ind, i in enumerate(item):         #it gives result in index format
    print(ind+1, i)'''
'''#class
class nice:
    def verynice(self):
        return "add salt to chicken"
class dog(nice):
    def __init__(self, name, age):
        self.name= name
        self.age=age
    def bark(self):
        return "woof"
a=dog("aditya", 18)
print(type(a))
print(a.name, a.age)
print(a.bark())        #You can remove print and remove return and add print there that would wield same result
print(a.verynice())'''
'''import dog
dog.bark()'''
#or
'''from dog import bark
bark()'''
'''import sys
print("hello", sys.argv[1])'''
'''import argparse
parser=argparse.ArgumentParser(
    description="HEllo there boys, how we doing?"
)
parser.add_arguement('-c', '--condition', metavar='condition', required=True, choices={"good", "bad"}, help='how is he?')
args=parser.parse_args()
print(args.condition)                  #doubt
'''
'''c=lambda a,b : a+b        # any artimatic operators you can try
print(c(4,5))'''
'''b=[2,4,6]
def double(a):
    return a*2
res=map(double,b)
print(set(res))'''
#or
'''b=[2,4,6]
c=lambda a : a*2
res=map(c,b)
print(list(res))'''
#or
'''b=[2,4,6]
res=map(lambda a : a*2, b)
print(list(res))'''
'''a=["apple", "grnt", "shqrt", "jmnt", "flip"]
def fille(c):
    v=["a","e","i","o","u"]
    for i in v:
        return i in c
res=filter(fille, a)
print(list(res))
'''
#or 
'''a=[1,2,3,4,5,6]
print(list(filter(lambda n : n%2==0, a)))'''
#or
print(list(n for n in a if n%2==0))
'''from functools import reduce              #it's called reduce 'cause it narrows down to a single component
tl=[("a",60),("b",62)]
print(reduce(lambda a, b : a[1]+b[1], tl))'''
#reduce repeatedly applies the functions to the elements and returns a single value
#in reduce, lambda always takes two arguements
a=[1,2,3,4]
print(reduce(lambda x,y:x*y,a))
'''def factorial(n):
    if n==1 : return 1         # base case
    return n * factorial(n-1)   # recursive case
print(factorial(6))'''
'''def logtime(func):
    def wrapper():
        print("before")
        v=func()
        print("after")
        return v
    return wrapper            # decorator can be used in many things for egs. we can use to perform the same functions in different other functions without modifying the function itself
@logtime
def hello():
    print("hello")
hello()
'''
#for more info about docstrings look for that video

'''def number(a : 'int') -> 'int':
    return a
print(number("ana"))'''        #doubt

'''try:
    raise Exception("you're onto something, aren't you?")
    a=2/0
except Exception as error:
    print(error)
finally:
    a=1
print(a)'''
'''class adinotfoundexcepton(Exception):
    print("inside")            #we can use pass statement to clarify that it's a function with nothing inside
    pass
try:
    raise adinotfoundexcepton("where are you, adi??")
except adinotfoundexcepton as error:
    print(error)
    print("he's gone, lol")'''
#with
try:                                  #we are try here because there could be exceptions
    file=open(filename.txt,"r")
    content=file.read()
    print(content)
finally:                               #finally we are making sure that the file is closed 
    file.close()
#alternate method
with open(filename.txt, "r") as file:          #using with, it will automatically close the file, in other word, with is a great way in exception handling as it automatically closes the file
    content=file.read()
    print(content)
#pip
'''pip install/uninstall requests
pip install -U requests     #update
pip show requests'''     #view info
'''with ("hello.txt", "r") as file:
    content=file.read()
    print(content)'''
'''l=[1,2,3,4,5]
n=[n**2 for n in l]
print(n)'''                #you can also do that using lambda function and map func as well
#polymorphism is very easy you can look upto it on the internet
class dog:
    def __init__(self,a,b):
        self.age=b
        self.name=a
    def __gt__(self, other):
        return true if self.age > other.age else False
q=dog("adi",7)
c=dog("bibhor", 18)
print(q>c)
#__add__()
#__sub__()
#__mul__()
#__truediv__()
#__floordiv__()
#__mod__()
#__pow__()
#__lshift__() <<
#__rshift__() >>
#__and__()
#__or__()
#__xor__()
