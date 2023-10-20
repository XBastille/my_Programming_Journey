import random
from dice_visual import dice_drawing
roll = input("Roll the dice? (Yes/No):")
while roll=="yes":
        dice1=random.randint(1, 6)
        dice2=random.randint(1, 6)
        print(f"dice rolled: {dice1} and {dice2}")
        print("\n".join(dice_drawing[dice1]))
        print("\n".join(dice_drawing[dice2]))
        print(f"total value: {dice1+dice2}")
        roll = input("Roll again? (Yes/no): ").lower()