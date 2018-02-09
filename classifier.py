#/usr/bin/python

# naive implementation of a naive bayesian classifier for learning tic-tac-toe
# 2018-02-09
# Shane Ryan


import numpy as numpy


# read in the tic-tac-toe board data and store as lines in a list
training_data = list()
with open("tic-tac-toe.data", "r") as training_data_file:
    for line in training_data_file:
        training_data.append(line)

# parse training data, separating each game into wins vs losses
winning_boards = list()
losing_boards = list()
for game_board in training_data:
    if "positive" in game_board:
        winning_boards.append(game_board)
    elif "negative" in game_board:
        losing_boards.append(game_board)
    else:
        raise ValueError("Training data is incomplete", game_board)

# verifying that all game boards are sorted
if len(training_data) != len(winning_boards) + len(losing_boards):
    raise ValueError("Training data not completely sorted")



