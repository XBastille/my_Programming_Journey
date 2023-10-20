from PyDictionary import PyDictionary

from time import sleep
dictionary=PyDictionary()
mean=input("Enter the word to find the meaning:").lower()
form=dictionary.meaning(mean)
for i in form:
    print("Type:")
    print(i)
    sleep(1)
    print()
    print("meaning:")
    for index,j in enumerate(form[i]):
        print(f"{index+1}: {j}")
    sleep()
    print()
