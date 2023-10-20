import random
a="yes"
while a=="yes":
    def get_choices():
        pc=input("Enter your choice:")
        p=["rock", "paper", "scissors"]
        cc=random.choice(p)
        choices={"player":pc,"computer":cc}
        return choices

    def check(player,computer):
        if player==computer:
            return  "Tie"
        elif player=="rock" and computer=="paper":
            return  "lose"
        elif player=="rock" and computer=="scissors":
            return  "win"
        elif player=="paper" and computer=="scissors":
            return  "lose"
        elif player=="paper" and computer=="rock":
            return  "win"
        elif player=="scissor" and computer=="rock":
            return  "lose"
        elif player=="scissor" and computer=="paper":
            return  "win"
    a=get_choices()
    print(a)
    b=check(a["player"],a["computer"])
    print(b)
    a=input("wanna play again?")
