import pygame
from Board import Board

pygame.init()
window = pygame.display.set_mode((600, 720))

board = Board()
moves = 0

# draw initial window
board.draw(window)

while not board.solved:
    pygame.time.delay(100)
    # check for player quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    color = board.next_move()
    board.change_color(color)
    moves += 1

    # print board
    board.draw(window)

# display solved board
board.draw(window)

print(moves)
pygame.time.delay(250)

pygame.quit()
