class Block:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.captured = False

    def to_string(self):
        print("(" + str(self.x) + ", " + str(self.y) + ") - " + str(self.color))

    def set_captured(self, captured):
        self.captured = captured

    def set_color(self, color):
        self.color = color

    def is_captured(self):
        return self.captured

    def has_left(self):
        return self.x > 0

    def has_top(self):
        return self.y > 0

    def has_right(self):
        return self.x < 13

    def has_bottom(self):
        return self.y < 13

