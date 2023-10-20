import random

y="yes"
while y=="yes":
    a=[0,1,2,3,4,5,6]
    d=["batting","bowling"]
    def get_choices_oe():
        p=input("odd or even?(type 'odd' or 'even')")
        b=int(input("make your move?(0-6):"))
        cc=random.choice(a)
        print("you chose:", b)
        print("computer chose:", cc)
        if p=="even" and (cc+b)%2==0:
            z=input("you won the toss! choose batting or bowling:")
            return "bowling" if z=="batting" else "batting"
        elif p=="odd" and (cc+b)%2==0:
            print("oh u lost the toss")
            x=random.choice(d)
            print("computer chose:", x)
            return x
        elif p=="odd" and (cc+b)%2!=0:
            z=input("you won the toss! choose batting or bowling:")
            print("you chose:", z)
            return "bowling" if z=="batting" else "batting"
        else:
            print("oh u lost the toss")
            x=random.choice(d)
            print("computer chose:", x)
            return xprint("oh u lost the toss")
            x=random.choice(d)
            print("computer chose:", x)
            return x
    def get_choices_game(o):
        co=0
        pl=0
        if o=="batting":
            print("start balling!!!")
            while o=="batting":
                e=int(input("make your move?(0-6):"))
                f=random.choice(a)
                print("u throw:", e)
                print("computer shots:", f)
                if e==f:
                    print("computer is out")
                    o="bowling"
                elif f==0:                            
                    co+=e 
                    print("your score:", pl ,"computer score:", co)
                else:
                    co+=f
                    print("your score:", pl ,"computer score:", co)
            print("scores so far. Your score:", pl , "computer score:", co)
            print("game starts again, start batting!!!")
            while o=="bowling" and pl<=co:
                e=int(input("make your move?(0-6):"))
                f=random.choice(a)
                print("u shot:", e)
                print("computer throws:", f)
                if e==f:
                    print("you are out")
                    o="xxx"
                elif e==0:
                    pl+=f
                    print("your score:", pl ,"computer score:", co)
                else:
                    pl+=e
                    print("your score:", pl ,"computer score:", co)
            return [co,pl]
        else:
            print("start batting!!!")
            while o=="bowling":
                e=int(input("make your move?(0-6):"))
                f=random.choice(a)
                print("u shot:", e)
                print("computer throws:", f)
                if e==f:
                    print("you are out")
                    o="batting"
                elif e==0:
                    pl+=f
                    print("your score:", pl ,"computer score:", co)
                else:
                    pl+=e
                    print("your score:", pl ,"computer score:", co)
            print("scores so far. Your score:", pl , "computer score:", co)
            print("game starts again, start bowling!!!")
            while o=="batting" and co<=pl:
                e=int(input("make your move?(0-6):"))
                f=random.choice(a)
                print("u throw:", e)
                print("computer shots:", f)
                if e==f:
                    print("computer is out")
                    o="xxx"
                elif f==0:
                    co+=e 
                    print("your score:", pl ,"computer score:", co)
                else:
                    co+=f
                    print("your score:", pl ,"computer score:", co)
            return [co,pl]
    def total_scores(r,t):
        print("final score:, COMPUTER SCORE:", r ,"YOUR SCORE:", t) 
        if r==t:
            return "match draw"
        elif r>t:
            return "computer wins the game"
        else:
            return "you won the match!!!!!!"
    i=get_choices_oe()
    l=get_choices_game(i)
    print(total_scores(l[0],l[1]))
    y=input("wanna play again?")
print("Thanks for playing!!")
