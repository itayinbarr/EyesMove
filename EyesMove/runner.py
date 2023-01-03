from back import *

# Run this file to start the software
# ---------------------------

# create_model()
# Introduction
print("***************************")
print("***************************")
print("Welcome to EyesMove - A dense neural network model I trained, to recognize if patient has opened/closed eyes.")
print("---------------------------")
print("---------------------------")

# You can use the input to load 1 or multiple windows
print("Now loading example eeg data...")
example = int(input('type a number between 1-3000 to pick a patient and see if eyes closed'))
use_model(example)