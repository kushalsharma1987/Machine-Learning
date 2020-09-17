# Kushal Sharma
# kushals@usc.edu
# HW 7 - Hidden Markov Model
# INF 552 Summer 2020

import numpy as np

# Viterbi algorithm takes in initial probability, emission probability model, transition probability model
# and the observation sequence. Returns the state with maximum value for the observation at each time stamp.
def viterbi_algorithm(initial_prob, emission_model, transition_model, emission_seq):

    D = 10
    emission_max_state = {}
    transition_multiplication = initial_prob
    t = 0
    for eve in emission_seq:
        t = t + 1
        emission_multiplication = {}
        # table multiplication operation between P(X) and P(E|X)
        for i in range(1, D + 1):
            emission_multiplication[i] = transition_multiplication[i] * emission_model[eve, i]
        # table projection operation P(E,X) => P(X)
        emission_max_state[t, eve] = max(emission_multiplication, key=emission_multiplication.get)

        transition_multiplication = {}
        for i in range(1, D + 1):
            transition_state_value = []
            for j in range(1, D + 1):
                # table multiplication operation between P(X(t-1)) and P(X(t))
                transition_state_value.append(emission_multiplication[j] * transition_model[i, j])
            # table projection operation P(X(t), X(t)) => P(X(t))
            transition_multiplication[i] = max(transition_state_value)

    return emission_max_state


def main():

    D = 10
    output = open('output.txt', 'w')
    initial_prob = {}
    for i in range(1, D+1):
        initial_prob[i] = 0.1
    initial_prob_array = np.array(list(initial_prob.values())).reshape(1, -1)
    print('initial probability:', np.shape(initial_prob_array), '\n', initial_prob_array)
    output.write('initial probability: ' + str(np.shape(initial_prob_array)) + '\n' + str(initial_prob_array) + '\n\n')

    # Populate Emission probability table
    emission_model = {}
    for i in range(0, D+2):
        for j in range(1, D+1):
            if (i == j-1) or (i == j) or (i == j+1):
                emission_model[i, j] = 0.33
            else:
                emission_model[i, j] = 0
    emission_array = np.array(list(emission_model.values())).reshape(-1, D)
    print('emission probability model:', np.shape(emission_array), '\n', emission_array)
    output.write('emission probability model: ' + str(np.shape(emission_array)) + '\n' + str(emission_array) + '\n\n')

    # Populate Transition probability table
    transition_model = {}
    for i in range(1, D+1):
        for j in range(1, D+1):
            if j == 1 and i == 2:
                transition_model[i, j] = 1
            elif j == 10 and i == 9:
                transition_model[i, j] = 1
            elif i == j-1 or i == j+1:
                transition_model[i, j] = 0.5
            else:
                transition_model[i, j] = 0
    transition_array = np.array(list(transition_model.values())).reshape(-1, D)
    print('transition probability model:', np.shape(transition_array), '\n', transition_array)
    output.write('transition probability model: ' + str(np.shape(transition_array)) + '\n' + str(transition_array) + '\n\n')

    # Read the observation sequence from the query file.
    query_file = open('query.txt', 'r').read()
    emission_seq = [int(x) for x in query_file.split(',')]
    print('emission sequence:', emission_seq)
    output.write('emission sequence: ' + str(len(emission_seq)) + '\n' + str(emission_seq) + '\n\n')
    # emission_seq = [8, 6, 4, 6, 5, 4, 5, 5, 7, 9]

    # Call viterbi algorithm function to get the most likely state path for given observation sequence.
    emission_max_state = viterbi_algorithm(initial_prob, emission_model, transition_model, emission_seq)
    print('Most Likely state path in format {timestamp, emission: state}:', '\n', emission_max_state)
    output.write('Most Likely state path in format {timestamp, emission: state}:' + '\n' + str(emission_max_state) + '\n\n')


if __name__ == '__main__':
    main()