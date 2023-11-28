import random
import game
import sys

# Author:				chrn (original by nneonneo)
# Date:				11.11.2016
# Description:			The logic of the AI to beat the game.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
moves = [UP, DOWN, LEFT, RIGHT]

def find_best_move(board):
    bestmove = find_best_move_heuristic_agent(board)
    # bestmove = find_best_move_random_agent()
    return bestmove

def find_best_move_random_agent():
    return random.choice(moves)

def find_best_move_heuristic_agent(board):
    boards = [execute_move(move, board) for move in moves]
    scores = [find_max_score(board) for board in boards]
    best_moves = [move for move, score in enumerate(scores) if score == max(scores)]
    return random.choice(best_moves)

def find_max_score(board):
    boards = [execute_move(move, board) for move in moves]
    scores = [score_board(new_board) if not board_equals(board, new_board) else 0 for new_board in boards]
    return max(scores)

def score_board(board):
    """
    A board with more empty (0) tiles is defined as having a higher score.
    Try to stay in the game for as long as possible.
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
    return  (newboard == board).all()  