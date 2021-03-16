import enum
import random


class Colors(enum.Enum):
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Red = (255, 0, 0)
    Blue = (0, 0, 255)
    Yellow = (255, 255, 0)
    Orange = (255, 165, 0)

    @staticmethod
    def random_color():
        return random.choice(list(Colors))
