#/usr/bin/python

# naive implementation of a naive bayesian classifier for predicting the
# outcome of a tic-tac-toe match given the game board
# 2018-02-09
# Shane Ryan


# function for calculating relevant probabilites given a board position
def generate_probability(board_position, probs):
    x=0
    o=1
    b=2   # constants for later indexing probability lists
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
    if len(state_freqs) > 2:
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
        next_pos = probs[position]['conditional-win'][gameboard[index]]
        print(next_pos)
        est_pos *= next_pos / \
                (probs[position]['conditional-win'][gameboard[index]]*\
                    total_prob['win'] + \
                 probs[position]['conditional-loss'][gameboard[index]]*\
                    total_prob['loss'])
        next_neg = probs[position]['conditional-loss'][gameboard[index]]
        print(next_neg)
        est_neg *= next_neg / \
                (probs[position]['conditional-win'][gameboard[index]]*\
                    total_prob['win'] + \
                 probs[position]['conditional-loss'][gameboard[index]]*\
                    total_prob['loss'])

    print(est_neg)
    print(est_pos)

    est_pos *= total_prob['win']
    est_neg *= total_prob['loss']

    print(total_prob)

    print(est_pos)
    print(est_neg)

    if est_pos > est_neg:
        return "y"
    elif est_pos < est_neg:
        return "n"
    elif est_pos == est_neg:
        raise ValueError("Equally probable outcomes!")
        return
    else:
        raise ValueError("Error finding MAP estimator")
        return


# parse training data, separating each game into wins vs losses, removing
# spaces in data and storing only the nine values corresponding to state of
# top left, top center, top right, etc.
def parse_wins_losses(data):
    winning_boards = list()
    losing_boards = list()

    for game_board in data:
        shortened = game_board.replace(" ","")
        if (shortened[4] == 'y'):
            winning_boards.append(game_board.replace(" ", "")[0:4])
        elif (shortened[4] == 'n'):
            losing_boards.append(game_board.replace(" ", "")[0:4])
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
        scenario.append(shortened[0:4])
        scenario.append(shortened.rstrip()[4:])
        outcomes.append(scenario)
    return outcomes

########################### program start #####################################

# read in the tic-tac-toe board data and store as lines in a list
training_data = read_data("weather.data")
test_data = parse_test_data(read_data("weather.test"))

print(test_data)

# parse wins / losses from training data
winning_boards, losing_boards = parse_wins_losses(training_data)

print(winning_boards)
print(losing_boards)

## develop dictionary of key / val pairs for frequency tables

# each position on the table has a corresponding location in the training data.
# in this case, our training data starts at the top left of the board, and
# moves through the three rows.  each board position thus has a value for
# indexing the training data correctly and two dictionaries to store win / loss

training_freqs = {'top_left' : [0, {'win':{'s' : 0, 'o' : 0, 'r' : 0},
                                    'loss':{'s' : 0, 'o' : 0, 'r' : 0}}],

                'top_center' : [1, {'win':{'h' : 0, 'm' : 0, 'c' : 0},
                                    'loss':{'h' : 0, 'm' : 0, 'c' : 0}}],

                'top_right' : [2, {'win':{'h' : 0, 'n' : 0},
                                    'loss':{'h' : 0, 'n' : 0}}],

                'middle_left' : [3, {'win':{'f' : 0, 't' : 0},
                                    'loss':{'f' : 0, 't' : 0}}],

#                 'middle_center' : [4, {'win':{'x' : 0, 'o' : 0,'b':0},
#                                     'loss':{'x' : 0, 'o' : 0, 'b': 0}}],
#
#                 'middle_right' : [5, {'win':{'x' : 0, 'o' : 0, 'b':0},
#                                     'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],
#
#                 'bottom_left' : [6, {'win':{'x' : 0, 'o' : 0, 'b' :0},
#                                     'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],
#
#                 'bottom_center' : [7, {'win':{'x' : 0, 'o' : 0,'b':0},
#                                     'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],
#
#                 'bottom_right' : [8, {'win':{'x' : 0, 'o' : 0, 'b':0},
#                                     'loss':{'x' : 0, 'o' : 0, 'b' : 0}}],
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

print(training_freqs)

# calculate conditional probabilites for each board position
state_prob = {}
prob_num = 0
probs1 = {'x':0, 'o':0, 'b':0,
         'conditional-win': {'s':0, 'o':0, 'r':0},
         'conditional-loss':{'s':0, 'o':0, 'r':0}}
probs2 = {'x':0, 'o':0, 'b':0,
         'conditional-win': {'h':0, 'm':0, 'c':0},
         'conditional-loss':{'h':0, 'm':0, 'c':0}}
probs3 = {'x':0, 'o':0, 'b':0,
         'conditional-win': {'h':0, 'n':0},
         'conditional-loss':{'h':0, 'n':0}}
probs4 = {'x':0, 'o':0, 'b':0,
         'conditional-win': {'t':0, 'f':0},
         'conditional-loss':{'t':0, 'f':0}}
list_probs = [probs1, probs2, probs3, probs4]

for position in training_freqs:
    state_prob[position] = generate_probability(position, list_probs[prob_num])
    state_prob[position]['index'] = training_freqs[position][0]
    prob_num += 1

# add in the overall probability of win vs. loss
total_prob = {}
total_prob['win'] = len(winning_boards) / (len(winning_boards) + \
        len(losing_boards))
total_prob['loss'] = len(losing_boards) / (len(winning_boards) + \
        len(losing_boards))

print(winning_boards)
print(winning_boards)

for position in state_prob:
    for line in state_prob[position]['conditional-win']:
        print(state_prob[position]['conditional-win'][line])
        print(state_prob[position]['conditional-loss'][line])

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


