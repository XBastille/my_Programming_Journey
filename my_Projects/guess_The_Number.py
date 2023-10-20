import random
from time import sleep as death    #because sleep is the cousin of death
def range_of_guessing():
    guess_start=-1
    while guess_start<0:
        try:
            guess_start=int(input("select the starting point of range for guessing :"))
        except Exception:
            print("umm Bro, tell me the starting point of the range for guessing")
    guess_end=-1
    while guess_end<=0:
        try:
            guess_end=int(input("select the ending point of range for guessing :"))
        except Exception:
            print("umm Bro, tell me the ending point of the range for guessing")
    death(1)
    return guess_start, guess_end
def guessing_time(guess_start, guess_end):
    choice=input("who will think of a number?, (I/computer):").lower()
    while choice not in ("i", "computer", "comp.", "comp", "me"):
        choice=input("Umm bro, tell me who the hell will guess (I/Computer)?:").lower()
    print("Let the game begin")
    death(1)
    if choice in ("computer", "comp.", "comp"):
        guess_it(guess_start, guess_end)
    else:
        guess_it2(guess_start, guess_end)
def guess_it2(guess_start, guess_end):
    print("Think about the number that u want the computer to guess!!")
    print()
    death(5)
    thinking=True
    while thinking:
        ti=input("are you done thinking?:").lower()
        while ti not in ("yes", "y"):
            ti=input("are you done thinking?:").lower()
        if ti in ("yes", "y"):
            print()
            print("okay then!!")
            thinking=False
    guessing=True
    round=0
    while guessing:
        nora=random.randint(guess_start, guess_end)
        check=input(f"is {nora} the number that u thought?:").lower()
        while check not in ("yes", "no", "n", "no"):
            check=input(f"umm BRO!,is {nora} the number that you thought?? write(yes/no/y/n)??:").lower()
        if check in ("no", "n"):
            print()
            death(1)
            h=input("was the guess higher?, a bit higher?, way higher? or lower?, a bit lower?, way lower?:").lower()
            while h not in ("bit higher", "a bit higher", "higher", "way higher", "bit lower", "a bit lower", "lower", "way lower"):
                h=input("umm broo, tell me was guess higher?, a bit higher?, way higher? or lower?, a bit lower?, way lower?:").lower()
            print()
            if h in ("bit higher", "a bit higher"):
                guess_end=nora-1
                if round==0:
                    guess_start=nora-9
                round+=1
            elif h=="higher":
                round=0
                guess_end=nora-1
                guess_start=nora-29
            elif h=="way higher":
                round=0
                guess_end=nora-29
            elif h in ("bit lower", "a bit lower"):
                guess_start=nora+1
                if round==0:
                    guess_end=nora+9
                round+=1
            elif h=="lower":
                round=0
                guess_start=nora+1
                guess_end=nora+29
            elif h=="way lower":
                round=0
                guess_start=nora+29
        elif check in ("yes", "y"):
            print("there we go computer has  guessed your number!!")
            guessing=False
def guess_it(guess_start, guess_end):
    no=random.randint(guess_start, guess_end)
    print("computer chose a number from the given range!")
    display("Now, it's time to guess the number!!")
    print()
    guessing=True
    round1=0
    while guessing:
        nos=0
        while nos<=0:
            try:
                nos=int(input("Guess the number (within the range):"))
            except Exception:
                print("umm, bro guess the number please!!")
        if nos<guess_start or nos>guess_end:
            print("umm bro, guess within the said range!!")
        elif nos>no:
            if nos-no in range(1,10):
                print("U guessed a little higher")
            elif nos-no in range(10, 30):
                print("You guessed it higher!")
            else:
                print("U guessed it wayy higher!!")
        elif nos<no:
            if no-nos in range(1,10):
                print("U guessed a little lower")
            elif no-nos in range(10, 30):
                print("You guessed it lower!")
            else:
                print("U guessed it wayy lower!!")
        else:
            print("Bravo, U guessed it right!!!")  
            break
        if round1>=7:
            give_up=input("Do u wanna give up??:").lower()
            while give_up not in ("yes", "no", "n", "y"):
                give_up=input("umm bro,I asked whether u wanna give or not, say(yes/no/y/n):").lower()
            if give_up in ("yes", "y"):
                print()
                print(f"okay bro, the number was {no}, better luck next time")
                break
            else:
                display("cool")
        round1+=1
def display(arg0):
    death(1)
    print()
    print(arg0)
print("welcome to  guess game, here either u or computer will guess the number!! based on what u wish!!")
death(1)
print()
t=range_of_guessing()
guessing_time(*t)
