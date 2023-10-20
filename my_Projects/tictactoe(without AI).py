y="yes"
while y=="yes":
    import time
    import random
    slot={"A1":"_", "A2":"_", "A3":"_", "B1":"_","B2":"_", "B3":"_", "C1":"_", "C2":"_", "C3":"_"}
    class selection: 
        def choice(self):
            pc=input("select your choice cross/circle?:").lower()
            while pc not in ["circle", "cross"]:
                pc=input("umm bro, tell me what u want?, cross or circle?:").lower()
            return pc
        def first_hit(self):
            if random.randint(0,1)==0:
                return self.represent2("computer will hit first", "cc")
            else:
                return self.represent2("you will hit first", "pc")
        def represent2(self, arg0, arg1):
            time.sleep(1)
            print(arg0)
            return arg1
    class matrices:
        def slotting(self, mark, object, turn):
            map={1:"A1", 2:"A2", 3:"A3", 4:"B1", 5:"B2", 6:"B3", 7:"C1", 8:"C2", 9:"C3"}
            if slot[map[mark]] not in ["X", "O"]:
                slot[map[mark]] = "X" if object=="cross" else "O"
                return False
            else:
                if turn=="pc":
                    print(f"umm, bro the {mark}th position is already filled")
                return True
        def show_matrix(self):
            return f'''[{slot["A1"]} {slot["A2"]} {slot["A3"]}]
[{slot["B1"]} {slot["B2"]} {slot["B3"]}]
[{slot["C1"]} {slot["C2"]} {slot["C3"]}]'''
        def check_matrix_full(self):
            for i in list(slot.values()):
                if i=="_":
                    return True
            print()
            time.sleep(1)
            print("Oh no, it's a draw!!! *sob* *sob*")
            return False
    class gameplay:
        def __init__(self, turn):
            self.turn=turn
        def check_winner(self, object):
            if slot["A1"]==slot["A2"]==slot["A3"]==object or slot["A1"]==slot["B1"]==slot["C1"]==object:
                if self.turn=="pc":
                    self.represent("boiii, you won less go !!!")
                else:
                    self.represent("boi, you lost *sad noises*")
                return False
            elif slot["B1"]==slot["B2"]==slot["B3"]==object or slot["A2"]==slot["B2"]==slot["C2"]==object:
                if self.turn=="pc":
                    self.represent("boiii, you won less go !!!")
                else:
                    self.represent("boi, you lost *sad noises*")
                return False
            elif slot["C1"]==slot["C2"]==slot["C3"]==object or slot["A3"]==slot["B3"]==slot["C3"]==object:
                if self.turn=="pc":
                    self.represent("boiii, you won less go !!!")
                else:
                    self.represent("boi, you lost *sad noises*")
                return False
            elif slot["A1"]==slot["B2"]==slot["C3"]==object or slot["A3"]==slot["B2"]==slot["C1"]==object:
                if self.turn=="pc":
                    self.represent("boiii, you won less go !!!")
                else:
                    self.represent("boi, you lost *sad noises*")
                return False
            return True
        def represent(self, arg0):
            print()
            time.sleep(1)
            print(arg0)
        def play(self):  
            sl=selection()
            print()
            objectpc=sl.choice()
            objectcc = "circle" if objectpc=="cross" else "cross"
            playing=True
            round=0
            while playing:
                m=matrices()
                if self.turn=="pc":
                    if round>0:
                        print("Now, it's your turn!!!")
                    mark=0
                    nn=True
                    print()
                    time.sleep(1)
                    while mark<=0 or mark>9:
                        try:
                            mark=int(input("enter a no. to fill a spot!!:"))
                        except ValueError:
                            print("brruh, write the correct input")
                    while nn:
                        if mark>0 or mark<=9:
                            nn=m.slotting(mark, objectpc, self.turn)
                        if nn:
                            try:
                                mark=int(input("enter a no. to fill a spot!!:"))
                            except ValueError:
                                print("brruh, write the correct input")
                        if mark<=0 or mark>9:
                            nn=True
                    print()
                    time.sleep(1)
                    print(m.show_matrix())
                    print()
                    time.sleep(1)
                    if (
                        objectpc == "cross"
                        and not self.check_winner("X")
                        or objectpc != "cross"
                        and not self.check_winner("O")
                    ):
                        break
                    if not m.check_matrix_full():
                        break
                    self.turn="cc"
                    round+=1
                if self.turn=="cc":
                    if round>0:
                        print("Now, it's computer turn!!!")
                    print()
                    time.sleep(1)
                    op=True
                    while op:
                        mark=random.randint(1,9)
                        op=m.slotting(mark, objectcc, self.turn)
                    print(m.show_matrix())
                    print()
                    time.sleep(1)
                    if (
                        objectcc == "cross"
                        and not self.check_winner("X")
                        or objectcc != "cross"
                        and not self.check_winner("O")
                    ):
                        break
                    if not m.check_matrix_full():
                        break
                    self.turn="pc"
                    round+=1
    print("Hello there, Welcome to Tic_Tac_Toe!!!")
    time.sleep(1)
    print()
    print("RULES:-")
    print()
    print('''[1 2 3]
[4 5 6]
[7 8 9]''')
    time.sleep(1)
    print()
    print("the above figure shows the position in the matrix to slot your symbol, just type the number that is shown above to slot your symbol!!!, simple!!")
    time.sleep(5)
    print()
    sll=selection()
    g=gameplay(sll.first_hit())
    g.play()
    y=input("do u play wanna again?").lower()
print("thanks for playing boiii!!!!")


                        
        

