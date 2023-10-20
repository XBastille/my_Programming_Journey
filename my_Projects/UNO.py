import random
import time
color = ('RED','GREEN','BLUE','YELLOW')
rank = ('0','1','2','3','4','5','6','7','8','9','Skip','Reverse','Draw2','Draw4','Wild')
ctype = {'0':'number','1':'number','2':'number','3':'number','4':'number','5':'number','6':'number',
            '7':'number','8':'number','9':'number','Skip':'action','Reverse':'action','Draw2':'action',
            'Draw4':'action_nocolor','Wild':'action_nocolor'}
class Card:
    def __init__(self, color, rank):
        self.rank = rank
        if ctype[rank]=="number":
            self.color=color
            self.cardtype="number"
        elif ctype[rank]=="action":
            self.color = color
            self.cardtype="action"
        else:
            self.color = None
            self.cardtype="action_nocolor"
    def __str__(self):
        return self.rank if self.color is None else f"{self.color} {self.rank}"
class Deck:
    def __init__(self):
        self.deck = []
        for clr in color:
            for ran in rank:
                if ctype[ran]!='action_nocolor':
                    self.deck.append(Card(clr, ran))
                    self.deck.append(Card(clr, ran))
                else:
                    self.deck.append(Card(clr, ran))
    def __str__(self):
        deck_comp = "".join('\n' + card.__str__() for card in self.deck)
        return f'The deck has {deck_comp}'
    def shuffle(self):
        random.shuffle(self.deck)
    def deal(self):
        return self.deck.pop()
class Hand:
    def __init__(self):
        self.cards=[]
        self.cardsstr=[]
        self.number_cards=0
        self.action_cards=0
    def add_card(self, card):
        self.cards.append(card)
        self.cardsstr.append(str(card))
        if card.cardtype=="number":
            self.number_cards+=1
        else:
            self.action_cards+=1
    def remove_card(self, place):
        self.cardsstr.pop(place - 1)
        return self.cards.pop(place - 1)
    def cards_in_hand(self):
        for i in range(len(self.cardsstr)):
            print(f' {i + 1}.{self.cardsstr[i]}')
    def single_card(self, place):
        return self.cards[place - 1]
    def no_of_cards(self):
        return len(self.cards)
def choose_first():
    return "Player" if random.randint(0,1)==0 else "Computer"
'''Function to check if the card thrown by Player/computer is a valid card by comparing it with the top card'''
def single_card_check(top_card,card):
    return (
        card.color == top_card.color
        or top_card.rank == card.rank
        or card.cardtype == 'action_nocolor'
    )
'''FOR computer ONLY'''
'''To check if computer has any valid card to throw''' 
def full_hand_check(hand,top_card):
    return next(
        (
            hand.remove_card(hand.cardsstr.index(str(c)) + 1)
            for c in hand.cards
            if c.color == top_card.color
            or c.rank == top_card.rank
            or c.cardtype == "action_nocolor"
        ),
        "no card",
    )
'''Function to check if either wins'''
def win_check(hand):
    return len(hand.cards)==0
'''Function to check if last card is an action card (GAME MUST END WITH A NUMBER CARD)'''
def last_card_check(hand):
    for c in hand.cards:
        return c.cardtype!="number"
'''The gaming loop'''
while True:
    print('Welcome to UNO! Finish your cards first to win, less go boiii !!')
    deck = Deck()
    deck.shuffle()
    player_hand = Hand()
    for i in range(7):
        player_hand.add_card(deck.deal())
    pc_hand = Hand()
    for i in range(7):
        pc_hand.add_card(deck.deal())
    top_card = deck.deal()
    if top_card.cardtype!='number':
        while top_card.cardtype!='number':
            top_card = deck.deal()
    print(f"Starting Card is: {top_card}")
    time.sleep(1)
    playing = True
    turn=choose_first()
    print(f"{turn} will go first")
    pulled=0
    while playing:
        if turn=="Player":
            print(f"\nTop card is:  {str(top_card)}")
            print("Your cards: ")
            player_hand.cards_in_hand()
            if player_hand.no_of_cards()==1:
                if last_card_check(player_hand):
                    print('Last card cannot be action card \nAdding one card from deck')
                    player_hand.add_card(deck.deal())
                    print('Your cards: ')
                    player_hand.cards_in_hand()
            choice=input("\nHit or Pull?:").lower()
            while choice not in ["hit", "pull", "h", "p"]:
                choice=input("brruhh, pls write hit or pull or (h/p):").lower()
            if choice in ["hit", 'h']:
                pos=0
                if player_hand.no_of_cards()==2:
                    pos=input("Enter index of card: (UNO!) ").lower()
                    if pos!="uno!":
                        print()
                        time.sleep(1)
                        print("got ya!!!")
                        player_hand.add_card(deck.deal())
                        player_hand.add_card(deck.deal())
                        turn="Computer"
                pos=0
                while pos<=0:
                    try:
                        pos=int(input('Enter index of card:'))
                    except:
                        print("uhh, bro write the index pls hehe")
                temp_card=player_hand.single_card(pos)
                if single_card_check(top_card, temp_card):
                    if temp_card.cardtype=='number':
                        top_card = player_hand.remove_card(pos)
                        turn='Computer'
                    else:
                        if temp_card.rank=='Skip':
                            pulled=0
                            turn = 'Player'
                            top_card = player_hand.remove_card(pos)
                        elif temp_card.rank=='Reverse':
                            pulled=0
                            turn = 'Player'
                            top_card = player_hand.remove_card(pos)
                        elif temp_card.rank=='Draw2':
                            print("computer gets 2 cards, lmao")
                            pc_hand.add_card(deck.deal())
                            pc_hand.add_card(deck.deal())
                            pulled=0
                            time.sleep(1)
                            top_card = player_hand.remove_card(pos)
                            turn = 'Player'
                        elif temp_card.rank=='Draw4':
                            print("computer gets 4 cards, lmao")
                            for i in range(4):
                                pc_hand.add_card(deck.deal())
                            top_card = player_hand.remove_card(pos)
                            draw4color = input('Change color to:').upper()
                            while draw4color not in ("BLUE", "RED", "YELLOW", "GREEEN"):
                                draw4color=input("umm bro, say the color:")
                            pulled=0
                            top_card.color = draw4color
                            time.sleep(1)
                            turn = 'Player'
                        elif temp_card.rank=='Wild':
                            top_card = player_hand.remove_card(pos)
                            wildcolor = input('Change color to:').upper()
                            while wildcolor not in ("BLUE", "RED", "YELLOW", "GREEEN"):
                                wildcolor=input("umm bro, just say the color:")
                            top_card.color = wildcolor
                            turn = 'Computer'
                else:
                    print('This card cannot be used')
            elif choice in ('p','pull'):
                if pulled==0:
                    temp_card = deck.deal()
                    print(f"You got:  {str(temp_card)}")
                    time.sleep(1)
                    pulled+=1
                    if single_card_check(top_card, temp_card):
                        player_hand.add_card(temp_card)
                    else:
                        print('Cannot use this card')
                        player_hand.add_card(temp_card)
                        turn = 'Computer'
                else:
                    print("Bro, u have already pulled one card")
                    time.sleep(1)
            if win_check(player_hand):
                print('\Boi, U WON!!!')
                playing = False
                break
        if turn=='Computer':
            pulled=0
            if pc_hand.no_of_cards()==1:
                if last_card_check(pc_hand):
                    time.sleep(1)
                    print('Adding a card to computer hand')
                    pc_hand.add_card(deck.deal())
            temp_card = full_hand_check(pc_hand, top_card)
            time.sleep(1)
            if temp_card!='no card':
                if pc_hand.no_of_cards()==1:
                    time.sleep(1)
                    print()
                    print("Computer says: UNO!!!")
                print(f'computer throws: {temp_card}')
                time.sleep(1)
                if temp_card.cardtype=='number':
                    top_card = temp_card
                    turn = 'Player'
                else:
                    if temp_card.rank=='Skip':
                        turn = 'Computer'
                        top_card = temp_card
                    elif temp_card.rank=='Reverse':
                        turn = 'Computer'
                        top_card = temp_card
                    elif temp_card.rank=='Draw2':
                        print("Oh nah my boi, u getting 2 cards *sob *sob*")
                        player_hand.add_card(deck.deal())
                        player_hand.add_card(deck.deal())
                        top_card = temp_card
                        time.sleep(1)
                        turn = 'Computer'
                    elif temp_card.rank=='Draw4':
                        print("Oh nah my boi, u getting 4 cards *sob *sob*")
                        for i in range(4):
                            player_hand.add_card(deck.deal())
                        top_card = temp_card
                        draw4color = pc_hand.cards[0].color
                        print('Color changes to', draw4color)
                        top_card.color = draw4color
                        time.sleep(1)
                        turn = 'Computer'
                    elif temp_card.rank=='Wild':
                        top_card = temp_card
                        wildcolor = pc_hand.cards[0].color
                        print("Color changes to", wildcolor)
                        top_card.color = wildcolor
                        turn = 'Player'
            else:
                print('\ncomputer pulls a card from deck')
                time.sleep(1)
                temp_card=deck.deal()
                if single_card_check(top_card, temp_card):
                    print(f'computer throws: {temp_card}')
                    time.sleep(1)
                    if temp_card.cardtype=='number':
                        top_card = temp_card
                        turn='Player'
                    else:
                        if temp_card.rank=='Skip':
                            turn='Computer'
                            top_card = temp_card
                        elif temp_card.rank=='Reverse':
                            turn='Computer'
                            top_card = temp_card
                        elif temp_card.rank=='Draw2':
                            player_hand.add_card(deck.deal())
                            player_hand.add_card(deck.deal())
                            top_card = temp_card
                            turn = 'Computer'
                        elif temp_card.rank=='Draw4':
                            for i in range(4):
                                player_hand.add_card(deck.deal())
                            top_card = temp_card
                            draw4color = pc_hand.cards[0].color
                            print('Color changes to', draw4color)
                            top_card.color = draw4color
                            turn = 'Computer'
                        elif temp_card.rank=='Wild':
                            top_card = temp_card
                            wildcolor = pc_hand.cards[0].color
                            print('Color changes to', wildcolor)
                            top_card.color = wildcolor
                            turn = 'Player'
                else:
                    print('computer doesnt have a card')
                    time.sleep(1)
                    pc_hand.add_card(temp_card)
                    turn = 'Player'
            print(f'computer has {pc_hand.no_of_cards()} cards remaining')
            time.sleep(1)
            if win_check(pc_hand):
                print('computer WON, and you lose *sob* *sob*')
                playing=False
    new_game = input('Would you like to play again? (y/n)').lower()
    if new_game in ['y', "yes"]:
        continue
    else:
        print("Thanks for playing!!")
        break
