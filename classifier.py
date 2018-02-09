#/usr/bin/python

# naive implementation of a naive bayesian classifier for predicting the
# outcome of a tic-tac-toe match given the game board
# 2018-02-09
# Shane Ryan


# function for calculating relevant probabilites given a board position
def generate_probability(board_position):
    x=0
    o=1
    b=2   # constants for later indexing probability lists
    probs = {'x':0, 'o':0, 'b':0,
             'x-given-win':0, 'o-given-win':0, 'b-given-win':0,
             'x-given-loss':0, 'o-given-loss':0, 'b-given-loss':0}

    # frequency of each state occurring regardless of win / loss
    # followed by frequency of win occurring regardless of state
    # and loss ocurring regardless of state
    state_freqs = list()
    wins = 0
    losses = 0
    for state in training_freqs[board_position][1]['win']:
        curr_win = training_freqs[board_position][1]['win'][state]
        curr_loss = training_freqs[board_position][1]['loss'][state]
        wins += curr_win
        losses += curr_loss
        state_freqs.append(curr_win + curr_loss)

    win_freq = wins / (wins + losses)
    loss_freq = losses / (wins + losses)

    probs['x'] = state_freqs[x] / sum(state_freqs)
    probs['o'] = state_freqs[o] / sum(state_freqs)
    probs['b'] = state_freqs[b] / sum(state_freqs)

    probs['x-given-win'] = training_freqs[board_position][1]['win']['x'] \
            / wins
    probs['x-given-loss'] = training_freqs[board_position][1]['loss']['x'] \
            / losses
    probs['o-given-win'] = training_freqs[board_position][1]['win']['o'] \
            / wins
    probs['o-given-loss'] = training_freqs[board_position][1]['loss']['o'] \
            / losses
    probs['b-given-win'] = training_freqs[board_position][1]['win']['b'] \
            / wins
    probs['b-given-loss'] = training_freqs[board_position][1]['loss']['b'] \
            / losses

    # basic checks to verify probabilities add up to 1
    #print(probs)
    #print(probs['x-given-win'] + probs['o-given-win'] + probs['b-given-win'])
    #print(probs['x-given-loss'] + probs['o-given-loss'] + probs['b-given-loss'])
    #print(probs['x'] + probs['o'] + probs['b'])
    ## i would leave this as a programmatic check, but sometimes rounding sets
    ## off false positives, i.e. probs. summing to 0.99

    return probs


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

training_freqs = {'top_left' : [0, {'win':{'x' : 0, 'o' : 0, 'b' : 0},
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
    for board_position in training_freqs:
        board_index = training_freqs[board_position][0]
        training_freqs[board_position][1]['win'][game_board[board_index]] += 1

# populate losing frequencies
for game_board in losing_boards:
    for board_position in training_freqs:
        board_index = training_freqs[board_position][0]
        training_freqs[board_position][1]['loss'][game_board[board_index]] += 1

# calculate conditional probabilites for each board position
state_probs = {}
for position in training_freqs:
    state_probs[position] = generate_probability(position)

# add in the overall probability of win vs. loss
state_probs['win'] = len(winning_boards) / (len(winning_boards) + \
        len(losing_boards))
state_probs['loss'] = len(losing_boards) / (len(winning_boards) + \
        len(losing_boards))

