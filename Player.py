import Board
from copy import copy


class Player:
    def __init__(self, board):
        self.board = copy(board)
        self.alive = True
        self.moves = 0

    def is_solved(self):
        return self.board.solved

    def get_data(self):
        return self.board.static_branch()

    def draw_board(self, window):
        self.board.draw(window)