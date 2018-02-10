#/usr/bin/python

# naive implementation of a naive bayesian classifier for predicting the
# outcome of a tic-tac-toe match given the game board
# 2018-02-09
# Shane Ryan

import tables
import sys

# function for calculating relevant probabilites given a board position
def generate_probability(board_position):
    x=0
    o=1
    b=2   # constants for later indexing probability lists
    probs = {'x':0, 'o':0, 'b':0,
             'conditional-win': {'x':0, 'o':0, 'b':0},
             'conditional-loss':{'x':0, 'o':0, 'b':0}}

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

    for state in probs['conditional-win']:
        probs['conditional-win'][state] = training_freqs[board_position][1][
                'win'][state] / wins
        probs['conditional-loss'][state] = training_freqs[board_position][1][
                'loss'][state] / losses

    # basic checks to verify probabilities add up to 1

    # print(probs)
    # print(probs['conditional-win']['x'] + \
        # probs['conditional-win']['o'] + probs['conditional-win']['b'])
    # print(probs['conditional-loss']['x'] + probs['conditional-loss']['o'] + \
        # probs['conditional-loss']['b'])
    # print(probs['x'] + probs['o'] + probs['b'])
    ## i would leave this as a programmatic check, but sometimes rounding sets
    ## off false positives, i.e. probs. summing to 0.99

    return probs


def read_data(file_name = "tic-tac-toe.data"):
    data = list()
    with open(file_name, "r") as open_file:
        for line in open_file:
            data.append(line)
    return data


# function to calculate the most probable game outcome given the board and
# calculated probabilities from training data
def map_estimator(gameboard, probs, total_prob):
    est_pos = 1;
    est_neg = 1;

    for position in probs:
        index = probs[position]['index']
        print(gameboard[index])

        denominator = (probs[position]['conditional-win'][gameboard[index]] * \
                total_prob['win'] + \
                probs[position]['conditional-loss'][gameboard[index]] * \
                total_prob['loss'])

        est_pos *= probs[position]['conditional-win'][gameboard[index]] / \
            denominator
        est_neg *= probs[position]['conditional-loss'][gameboard[index]] / \
            denominator

    est_pos *= total_prob['win']
    est_neg *= total_prob['loss']

    print(est_pos)
    print(est_neg)

    if est_pos > est_neg:
        return "positive"
    elif est_pos < est_neg:
        return "negative"
    elif est_pos == est_neg:
        raise ValueError("Equally probable outcomes!")
        return
    else:
        raise ValueError("Error finding MAP estimator")
        return


# generate a frequency table to display conditional probabilites
def gen_frequency_table(prob_data):

    headings = ['Position', 'P(X|Pos)', 'P(O|Pos)', 'P(B|Pos)',
                'P(X|Neg)', 'P(O|Neg)', 'P(B|Neg)']

    data = list()
    for position in prob_data:
        data.append(
                [position, round(prob_data[position]['conditional-win']['x'],4),
                    round(prob_data[position]['conditional-win']['o'],4),
                    round(prob_data[position]['conditional-win']['b'],4),
                    round(prob_data[position]['conditional-loss']['x'],4),
                    round(prob_data[position]['conditional-loss']['o'],4),
                    round(prob_data[position]['conditional-loss']['b'],4),
                    ]
                )

    fields = [0, 1, 2, 3, 4, 5, 6]

    align = [('^', '<'), ('^', '^'), ('^', '<'), ('^', '^'), ('^', '>'),
	     ('^','^'), ('^','^')]

    tables.table(sys.stdout, data, fields, headings, align)

# parse training data, separating each game into wins vs losses, removing
# spaces in data and storing only the nine values corresponding to state of
# top left, top center, top right, etc.
def parse_wins_losses(data):
    winning_boards = list()
    losing_boards = list()

    for game_board in data:
        if "positive" in game_board:
            winning_boards.append(game_board.replace(" ", "")[0:9])
        elif "negative" in game_board:
            losing_boards.append(game_board.replace(" ", "")[0:9])
        else:
            raise ValueError("Training data is incomplete", game_board)

    # verifying that all game boards are sorted
    if len(training_data) != len(winning_boards) + len(losing_boards):
        raise ValueError("Training data not completely sorted")

    return winning_boards, losing_boards


# parse the test data, returning a list of lists to separate the gameboard from
# the expected outcome
def parse_test_data(data):
    outcomes = list()
    for line in data:
        scenario = list()
        shortened = line.replace(" ", "")
        scenario.append(shortened[0:9])
        scenario.append(shortened.rstrip()[9:])
        outcomes.append(scenario)
    return outcomes



########################### program start #####################################

# read in the tic-tac-toe board data and store as lines in a list
training_data = read_data()
test_data = parse_test_data(read_data("tic-tac-toe.test"))

print(test_data)

# parse wins / losses from training data
winning_boards, losing_boards = parse_wins_losses(training_data)


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
state_prob = {}
for position in training_freqs:
    state_prob[position] = generate_probability(position)
    state_prob[position]['index'] = training_freqs[position][0]

# add in the overall probability of win vs. loss
total_prob = {}
total_prob['win'] = len(winning_boards) / (len(winning_boards) + \
        len(losing_boards))
total_prob['loss'] = len(losing_boards) / (len(winning_boards) + \
        len(losing_boards))

for line in state_prob:
    print(state_prob[line])

# now predict outcomes of the tic-tac-toe test situations given and report the
# accurracy of the algorithm
success = 0
failure = 0
for gameboard, outcome in test_data:
    estimation = map_estimator(gameboard, state_prob, total_prob)
    # print("Expected: {}\nCalculated: {}\n\n".format(outcome, estimation))
    if outcome == estimation:
        print("Test successful!")
        success += 1
    else:
        print("Test Failed :(")
        failure += 1

print("Successes: {}\nFailures: {}".format(success, failure))
gen_frequency_table(state_prob)

