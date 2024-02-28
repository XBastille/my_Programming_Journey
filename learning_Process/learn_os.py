import os
'''print(os.name)        #gives you information on what platform you're runnning on. here we got "nt" meaning windows. why "nt"? because microsoft started calling it os "nt" many years ago
print(os.environ["TMP"])    #returns a dictionary of user's environmental variable. every time you use your computer, some environment variable are set. this can give a lot of important information such as no. of processors used, types of CPUs, the computer name etc
print(os.getenv("TMP"))     #finding a variable in os.environ is same as getenv. getenv is recommended as it doesn't give any error if the given variable is not found
print(os.getcwd())   #let us know what path we are currently in.
os.chdir("your path here")    #let us change the path'''
#os.mkdir("hola24.py")         #let us create a folder (NOTE:- it's not a file)
#path=r"C:\Users\ASUS\Desktop\myWork\hola33.py"        #you can make a path for your folder 
#os.mkdir(path)
#path=r"C:\Users\ASUS\Desktop\myWork\hola43.py\2023\02\09"   
#os.makedirs(path)         #let us make nested folders
'''os.remove("the file")      #let us delete the file
os.rmdir("your directory")  #NOTE:- it's a folder
os.rename("your old", "new name")   
os.startfile("file path")'''
#path=r"C:\Users\ASUS"
#for root, dirs, files in os.walk(path):       #this will gives all the sub-directories of the given path. NOTE:- we must have root, dirs, files in the loop statement
#   print(root)
'''print(os.path.basename(r"C:\Users\ASUS\Desktop\myWork\parallel_processing.py"))  
print(os.path.dirname(r"C:\Users\ASUS\Desktop\myWork\parallel_processing.py"))  
print(os.path.exists(r"C:\Users\ASUS\Desktop\myWork\parallel_processing.py"))    
print(os.path.isfile(r"C:\Users\ASUS\Desktop\myWork\parallel_processing.py"))    
print(os.path.isdir(r"C:\Users\ASUS\Desktop\myWork\parallel_processing.py"))    
dirname, fname=os.path.split(r"C:\Users\ASUS\Desktop\myWork\parallel_processing.py")   #splits the file and it's directory into the tuple
print(dirname)
print(fname)
print(os.path.join(r"C:\Users\ASUS\Desktop\myWork", "parallel_processing.py"))'''
path=r"C:"
for root, dirs, files in os.walk(path):       #this will gives all the sub-directories of the given path. NOTE:- we must have root, dirs, files in the loop statement
   print(root)



