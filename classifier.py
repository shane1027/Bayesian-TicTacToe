#/usr/bin/python

# naive implementation of a naive bayesian classifier for learning tic-tac-toe
# 2018-02-09
# Shane Ryan


import numpy as numpy


# parse the tic-tac-toe board data and store as lists
training_data = list()
with open("tic-tac-toe.data", "r") as training_data_file:
    for line in training_data_file:
        training_data.append(line)

print(training_data)
