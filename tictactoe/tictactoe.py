"""
Tic Tac Toe Player
"""

from enum import Flag
from itertools import count
import math
# from symbol import term
X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]







def player(board):
    """
    Returns player who has the next turn on a board.
    """
    X_count, O_count = 0, 0
    for h in board:
        for cell in h:
            X_count += (cell == X)
            O_count += (cell == O)
    return X if X_count == O_count else O
    # raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.append((i, j))
    return actions
    # raise NotImplementedError


def result(board, action): 
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board=copy.deepcopy(board)
    board[action[0]][action[1]] = player(board)
    return board
    # raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if (board[i][0] == board[i][1] == board[i][2]) and board[i][0] != EMPTY:
            return board[i][0]
        if (board[0][i] == board[1][i] == board[2][i]) and board[0][i] != EMPTY:
            return board[0][i]
    
    if (board[0][0] == board[1][1] == board[2][2]) and board[0][0] != EMPTY:
        return board[0][0]

    if (board[0][2] == board[1][1] == board[2][0]) and board[0][2] != EMPTY:
        return board[0][2] 

    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True
    
    for h in board:
        for cell in h:
            if cell == EMPTY:
                return False 
    
    return True



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    result = winner(board)
    if result == None:
        return 0
    else:
        return 1 if result == X else -1

    # raise NotImplementedError

import copy
def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    v, move = max_value(copy.deepcopy(board)) if player(board)==X else min_value(copy.deepcopy(board))
    return move

    # raise NotImplementedError


def min_value(board):
    if terminal(board):
        return utility(board), None
    v1 = 2
    move = None
    for action in actions(board):
        v2, m = max_value(result(board,action))#result的结果需要深拷贝；只关注最上层的move值
        if v2 <v1:
            v1, move = v2, action
    return v1, move

def max_value(board):
    if terminal(board):
        return utility(board), None
    v1 = -2
    move = None
    for action in actions(board):
        v2, m = min_value(result(board, action))
        if v2 > v1:
            v1, move = v2, action
    return v1, move
