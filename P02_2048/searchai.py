import copy
import random
import game
import sys
from multiprocessing import Pool
from numpy import ndarray

# Author:      chrn (original by nneonneo)
# Date:        11.11.2016
# Copyright:   Algorithm from https://github.com/nneonneo/2048-ai
# Description: The logic to beat the game. Based on expectimax algorithm.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
moves = [UP, DOWN, LEFT, RIGHT]

def find_best_move(board):
    """
    :type board: ndarray
    """
    result = move_scores(board)

    best_moves = [move for move, score in enumerate(result) if score == max(result)]
    return random.choice(best_moves)


def move_scores(board, depth=2):
    """
    Entry Point to score the first move.
    """

    states = [[] if board_equals(board, execute_move(move, board)) else [GameState(board, 1)] for move in moves]
    scores = [0, 0, 0, 0]

    pool = Pool(4)
    for d in range(depth):
        results = pool.map(score_move_tuple, [(states[move], move) for move in moves])
        # results =  parallel(delayed(score_move)(states[move], move) for move in moves)
        # results = [score_move(states[move], move) for move in moves]

        states, scores = zip(*results)
        assert len(states) == 4
        assert len(scores) == 4

    pool.close()
    pool.join()

    # TODO:
    # Implement the Expectimax Algorithm.
    # 1.) Start the recursion until it reach a certain depth
    # 2.) When you don't reach the last depth, get all possible board states and
    #     calculate their scores dependence of the probability this will occur. (recursively)
    # 3.) When you reach the leaf calculate the board score with your heuristic.

    return scores


def score_move_tuple(t):
    return score_move(*t)


def score_move(states, move):
    """
    :type states: list[GameState]
    :type move: int
    :rtype: list[GameState], float
    """
    new_states = [GameState(execute_move(move, state.board), state.probability) for state in states]
    inserted_2 = sum([score_speculative_insertions(state, value=2, probability=state.probability * 0.9) for state in new_states], [])
    inserted_4 = sum([score_speculative_insertions(state, value=4, probability=state.probability * 0.1) for state in new_states], [])
    new_new_states = inserted_2 + inserted_4
    if len(new_new_states) > 0:
        score = sum([state.score() * state.probability for state in new_new_states]) / len(new_new_states)
    else:
        score = 0

    return new_new_states, score


def score_speculative_insertions(game_state, value, probability):
    """
    :type game_state: GameState
    :type value: int
    :type probability: float
    :rtype: list[GameState]
    """
    new_states = [GameState(board, probability) for board in speculative_insertions(game_state.board, value)]
    return new_states


def speculative_insertions(board, value):
    """
    :type board: ndarray
    :type value: int
    :rtype: list[ndarray]
    """
    boards = []
    x = 0

    while x < 16:
        n = x
        for n in range(x, 16):
            if board[n // 4][n % 4] == 0:
                new_board = copy.deepcopy(board)
                new_board[n // 4][n % 4] = value
                boards.append(new_board)
                break
        x = n + 1

    return boards


def empty_tile_count(board):
    """
    :type board: ndarray
    :rtype: int
    """
    return [tile for row in board for tile in row].count(0)


def execute_move(move, board):
    """
    move and return the grid without a new random tile 
    It won't affect the state of the game in the browser.
    """

    if move == UP:
        return game.merge_up(board)
    elif move == DOWN:
        return game.merge_down(board)
    elif move == LEFT:
        return game.merge_left(board)
    elif move == RIGHT:
        return game.merge_right(board)
    else:
        sys.exit("No valid move")


def board_equals(board, newboard):
    """
    Check if two boards are equal
    """
    return (newboard == board).all()

# def func_star(a_b):
#     """
# 	Helper Method to split the programm in more processes.
# 	Needed to handle more than one parameter.
#     """
#     return score_toplevel_move(*a_b)


class GameState:

    def __init__(self, board, probability):
        """
        :type board: ndarray
        :type probability: float
        """
        self.board = board
        self.probability = probability

    def score(self):
        best_move_score = 0
        for move in moves:
            new_board = execute_move(move, self.board)
            score = empty_tile_count(new_board)
            if not board_equals(self.board, new_board) and score > best_move_score:
                best_move_score = score

        return best_move_score
