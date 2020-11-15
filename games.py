import numpy as np

# Game 1: The (im)balancing act game. There are no NE under SER.
payoff1 = np.array([[(4, 0), (3, 1), (2, 2)],
                    [(3, 1), (2, 2), (1, 3)],
                    [(2, 2), (1, 3), (0, 4)]])

# Game 2: The (im)balancing act game without M. There are no NE under SER.
payoff2 = np.array([[(4, 0),  (2, 2)],
                    [(2, 2),  (0, 4)]])

# Game 3: The (im)balancing act game without R. (L, M) is a pure NE under SER.
payoff3 = np.array([[(4, 0), (3, 1)],
                    [(3, 1), (2, 2)]])

# Game 4: A two action game. There are NE under SER for (L,L) and (M,M).
payoff4 = np.array([[(4, 1), (1, 2)],
                    [(3, 1), (3, 2)]])

# Game 5: A three action game. There are  NE under SER for (L,L), (M,M) and (R,R).
payoff5 = np.array([[(4, 1), (1, 2), (2, 1)],
                    [(3, 1), (3, 2), (1, 2)],
                    [(1, 2), (2, 1), (1, 3)]])


def u1(vector):
    """
    This function calculates the utility for agent 1.
    :param vector: The reward vector.
    :return: The utility for agent 1.
    """
    utility = vector[0] ** 2 + vector[1] ** 2
    return utility


def gradient_u1(vector):
    """
    This function returns the partial derivative for the two objectives for agent 1.
    :param vector: The reward vector.
    :return: An array of the two partial derivatives for agent 1.
    """
    dx = 2 * vector[0]
    dy = 2 * vector[1]
    return np.array([dx, dy])


def u2(vector):
    """
    This function calculates the utility for agent 2.
    :param vector: The reward vector.
    :return: The utility for agent 2.
    """
    utility = vector[0] * vector[1]
    return utility


def gradient_u2(vector):
    """
    This function returns the partial derivative for the two objectives for agent 2.
    :param vector: The reward vector.
    :return: An array of the two partial derivatives for agent 2.
    """
    dx = vector[1]
    dy = vector[0]
    return np.array([dx, dy])


def get_payoff_matrix(game):
    """
    This function will provide the correct payoff game based on the game we play.
    :param game: The current game.
    :return: A payoff matrix.
    """
    if game == 'game1':
        payoff_matrix = payoff1
    elif game == 'game2':
        payoff_matrix = payoff2
    elif game == 'game3':
        payoff_matrix = payoff3
    elif game == 'game4':
        payoff_matrix = payoff4
    elif game == 'game5':
        payoff_matrix = payoff5
    else:
        raise Exception("The provided game does not exist.")

    return payoff_matrix


def get_u_and_du(agent):
    if agent == 1:
        return u1, gradient_u1
    else:
        return u2, gradient_u2


def get_min_max(game):
    """
    This method will return the minimum and maximum utility for both agents in the given game.
    :param game: The game for which we want the minimum and maximum utilities possible.
    :return: The minimum and the maximum for both agents
    """
    if game == 'game1':
        min1 = 8
        max1 = 16
        min2 = 0
        max2 = 4
    elif game == 'game2':
        min1 = 8
        max1 = 16
        min2 = 0
        max2 = 4
    elif game == 'game3':
        min1 = 8
        max1 = 16
        min2 = 0
        max2 = 4
    elif game == 'game4':
        min1 = 5
        max1 = 17
        min2 = 2
        max2 = 6
    else:
        min1 = 5
        max1 = 17
        min2 = 2
        max2 = 6

    return min1, max1, min2, max2
