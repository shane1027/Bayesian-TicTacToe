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

# parse training data, separating each game into wins vs losses, removing
# spaces in data and storing only the nine values corresponding to state of
# top left, top center, top right, etc.
winning_boards = list()
losing_boards = list()
for game_board in training_data:
    if "positive" in game_board:
        winning_boards.append(game_board.replace(" ", "")[0:9])
    elif "negative" in game_board:
        losing_boards.append(game_board.replace(" ", "")[0:9])
    else:
        raise ValueError("Training data is incomplete", game_board)

# verifying that all game boards are sorted
if len(training_data) != len(winning_boards) + len(losing_boards):
    raise ValueError("Training data not completely sorted")

## develop dictionary of key / val pairs for frequency tables
# each position on the table has a corresponding location in the training data.
# in this case, our training data starts at the top left of the board, and
# moves through the three rows.  each board position thus has a value for
# indexing the training data correctly and two dictionaries to store win / loss

training_data_frequencies = {'top_left' : [0, {'win':{'x' : 0, 'o' : 0, 'b' : 0},
                                        'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'top_center' : [1, {'win':{'x' : 0, 'o' : 0, 'b' : 0},
                                          'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'top_right' : [2, {'win':{'x' : 0, 'o' : 0, 'b' : 0},
                                         'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'middle_left' : [3, {'win':{'x' : 0, 'o' : 0, 'b' :0},
                                           'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'middle_center' : [4, {'win':{'x' : 0, 'o' : 0,'b':0},
                                             'loss':{'x' : 0, 'o' : 0, 'b': 0}}],

                            'middle_right' : [5, {'win':{'x' : 0, 'o' : 0, 'b':0},
                                            'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'bottom_left' : [6, {'win':{'x' : 0, 'o' : 0, 'b' :0},
                                            'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'bottom_center' : [7, {'win':{'x' : 0, 'o' : 0,'b':0},
                                             'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],

                            'bottom_right' : [8, {'win':{'x' : 0, 'o' : 0, 'b':0},
                                            'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],
                            }
# populate winning frequencies
for game_board in winning_boards:
    for board_position in training_data_frequencies:
        board_index = training_data_frequencies[board_position][0]
        training_data_frequencies[board_position][1]['win'][game_board[board_index]] += 1

# populate losing frequencies
for game_board in losing_boards:
    for board_position in training_data_frequencies:
        board_index = training_data_frequencies[board_position][0]
        training_data_frequencies[board_position][1]['loss'][game_board[board_index]] += 1


print(training_data_frequencies)



