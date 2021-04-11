from Block import Block
from Colors import Colors
from copy import copy, deepcopy
from numba import jit
import pygame
import numpy


class Board:
    def __init__(self, size):
        self.board = numpy.array([[Block(i, j, Colors.random_color().value, size) for i in range(size)] for j in range(size)])
        self.size = size
        self.board[0][0].set_captured(True)
        self.solved = False
        self.perimeter = 4
        self.moves = []
        self.change_color(self.board[0][0].color)
        self.captured = self.count_captured()

    def backtrack_util(self, max_depth, width):
        move_list = self.backtrack(0, max_depth, width)
        culled = move_list[0]
        for _ in range(len(self.moves)):
            culled.pop(0)
        return culled

    # finds the max captured of tree, and the move needed to get there
    def backtrack(self, depth, max_depth, width):
        # base case
        if depth >= max_depth or self.solved:
            result = (self.moves, self.captured)
            return result
        # else
        child_nodes = self.get_best_moves(width)
        max_captured = self.captured
        move_lists = []
        values = []
        for node in child_nodes:
            child_board = deepcopy(self)
            child_board.change_color(node[0])
            result = child_board.backtrack(depth + 1, max_depth, width)
            move_lists.append(list(result[0]))
            values.append(result[1])
        ind = values.index(max(values))
        ret = (move_lists[ind], values[ind])
        return ret

        # values = []
        # for node in child_nodes:
        #     child_board = deepcopy(self)
        #     child_board.change_color(node[0])
        #     values.append((node[0], child_board.backtrack(depth + 1, max_depth, n)))
        # return values

    def get_best_moves(self, n):
        options = []
        weights = self.branch()
        for index, entry in enumerate(weights):
            if index >= n or entry[1] == 0:
                break
            options.append(entry)
        return options

    def change_color(self, color):
        self.moves.append(color)
        perimeter = 0
        count = 0  # used to check if solved in same iteration
        checked = list()  # prevents the same block being checked twice
        temp = copy(self)  # copy needed to prevent multiple moves in a single method call
        for x in range(self.size):
            for y in range(self.size):
                block = temp.board[y][x]
                if block.is_captured():
                    count += 1
                    self.board[block.y][block.x].set_color(color)
                    if block.has_left():
                        if not (block.x - 1, block.y) in checked and not temp.get_left(block).captured and temp.get_left(block).color == color:
                            checked.append((block.x - 1, block.y))
                            self.recursive_capture(self.get_left(block), color)
                    else:
                        perimeter += 1
                    if block.has_top():
                        if not (block.x, block.y - 1) in checked and not temp.get_top(block).captured and temp.get_top(block).color == color:
                            checked.append((block.x, block.y - 1))
                            self.recursive_capture(self.get_top(block), color)
                    else:
                        perimeter += 1
                    if block.has_right():
                        if not (block.x + 1, block.y) in checked and not temp.get_right(block).captured and temp.get_right(block).color == color:
                            checked.append((block.x + 1, block.y))
                            self.recursive_capture(self.get_right(block), color)
                    else:
                        perimeter += 1
                    if block.has_bottom():
                        if not (block.x, block.y + 1) in checked and not temp.get_bottom(block).captured and temp.get_bottom(block).color == color:
                            checked.append((block.x, block.y + 1))
                            self.recursive_capture(self.get_bottom(block), color)
                    else:
                        perimeter += 1

        self.perimeter = perimeter

        self.captured = count
        if count == self.size * self.size:
            self.solved = True

    def count_captured(self):
        captured = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[y][x].captured:
                    captured += 1
        return captured

    def recursive_capture(self, block, color):
        block.set_captured(True)
        block.set_color(color)
        if block.has_left() and not self.get_left(block).captured and self.get_left(block).color == color:
            self.recursive_capture(self.get_left(block), color)
        if block.has_top() and not self.get_top(block).captured and self.get_top(block).color == color:
            self.recursive_capture(self.get_top(block), color)
        if block.has_right() and not self.get_right(block).captured and self.get_right(block).color == color:
            self.recursive_capture(self.get_right(block), color)
        if block.has_bottom() and not self.get_bottom(block).captured and self.get_bottom(block).color == color:
            self.recursive_capture(self.get_bottom(block), color)

    def get_left(self, block):
        if block.x == 0:
            return None
        x = block.x - 1
        y = block.y
        return self.board[y][x]

    def get_top(self, block):
        if block.y == 0:
            return None
        x = block.x
        y = block.y - 1
        return self.board[y][x]

    def get_right(self, block):
        if block.x == self.size - 1:
            return None
        x = block.x + 1
        y = block.y
        return self.board[y][x]

    def get_bottom(self, block):
        if block.y == self.size - 1:
            return None
        x = block.x
        y = block.y + 1
        return self.board[y][x]

    def draw(self, window):
        pygame.draw.rect(window, (60, 60, 60), window.get_rect())
        for row in self.board:
            for block in row:
                pygame.draw.rect(window, block.color, (block.x * 40 + 10, block.y * 40 + 10, 40, 40))
        pygame.display.update()

    # todo: turn into weight function for machine learning
    # returns the color choice which would capture the most blocks in the next move
    def next_move(self):
        return self.branch()[0][0]

    def branch(self):
        move_evals = dict((color.value, 0) for color in list(Colors))

        for color in list(Colors):
            temp = deepcopy(self)
            if color.value == temp.board[0][0].color:
                continue
            temp.change_color(color.value)
            move_evals[color.value] = temp.count_captured() - self.count_captured()
        move_evals = sorted(move_evals.items(), key=lambda item: item[1], reverse=True)
        return move_evals

    def static_branch(self):
        move_evals = [0, 0, 0, 0, 0, 0]
        for index, color in enumerate(list(Colors)):
            temp = deepcopy(self)
            if color.value == temp.board[0][0].color:
                continue
            temp.change_color(color.value)
            move_evals[index] = temp.count_captured() - self.count_captured()

        return move_evals

    # todo: turn into weight function for machine learning
    # returns the color which occurs most in the set of uncaptured blocks
    @jit(nopython=True)
    def most_remaining_move(self):
        # map used to count occurrences of colors in uncaptured blocks
        uncaptured_counts = dict((color.value, 0) for color in list(Colors))
        for row in self.board:
            for block in row:
                if not block.captured and not block.color == self.board[0][0].color:
                    uncaptured_counts[block.color] += 1
        uncaptured_counts = sorted(uncaptured_counts.items(), key=lambda item: item[1], reverse=True)
        return list(uncaptured_counts)[0][0]

    # todo: turn into weight function for machine learning
    # returns a random color which is not the current color
    def new_random_move(self):
        color = Colors.random_color().value
        while color == self.board[0][0].color:
            color = Colors.random_color().value
        return color
