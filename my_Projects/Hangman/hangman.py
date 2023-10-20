import random
from time import sleep 
from words_for_hangman import words
from hangman_visual import lives_visual_dict
import string
def get_valid_word(words):
    word = random.choice(words)  # randomly chooses something from the list
    while '-' in word or ' ' in word:
        word = random.choice(words)
    return word.upper()
def hangman():
    word = get_valid_word(words)
    word_letters=set(word)
    alphabet = set(string.ascii_uppercase)
    used_letters=set()
    lives=8
    count=0
    while word_letters and lives > 0:
        sleep(1)
        if count==0:
            print(f"You have {lives} lives left")
        else:
            print(f"You have {lives} lives left and you have used these letters: ", ' '.join(used_letters))
        count+=1
        word_list=[letter if letter in used_letters else '-' for letter in word]
        duplicate(lives)
        print("Current word: ", ' '.join(word_list))
        user_letter=input('Guess a letter: ').upper()
        if user_letter in alphabet - used_letters:
            used_letters.add(user_letter)
            if user_letter in word_letters:
                word_letters.remove(user_letter)
                print('')
            else:
                lives-=1
                sleep(1)
                print(f"\nYour letter, {user_letter} is not in the word.")
        elif user_letter in used_letters:
            sleep(1)
            print("\nYou have already used that letter, bro. Guess another letter.")
        else:
            sleep(1)
            print("\nThat is not a valid letter,man.")
    duplicate(lives)
    if lives == 0:
        print(f"Bro, you died, sorry. The word was {word}")
    sleep(1)
    if lives in range(5,7):
        print(f"Boi! You guessed the word {word} in {lives} lives !!, You win a gold award !!")
        print(lives_visual_dict[9])
    elif lives in range(3,5):
        print(f"Boi! You guessed the word {word} in {lives} lives !!, You win a silver medal !!")
        print(lives_visual_dict[9])
    elif lives in range(1,3):
        print(f"Boi! You guessed the word {word} in {lives} lives !!, You win a broze medal!!")
        print(lives_visual_dict[9])
def duplicate(lives):
    sleep(1)
    print(lives_visual_dict[lives])
    sleep(1)
if __name__ == '__main__':
    hangman()