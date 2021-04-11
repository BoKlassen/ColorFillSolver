from Board import Board
from copy import copy


def __init__(self, board):
    boardcopy = copy(board)
    color = board.board[0][0].color
    captured = board.captured
    perimeter = board.perimeter()
