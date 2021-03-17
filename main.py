import numba
import pygame
import math
import sys
import neat
from Board import Board
from Player import Player
from Colors import Colors

global generation
best_game = -1

def solve_new_board():
    # pygame and window
    pygame.init()
    window = pygame.display.set_mode((600, 720))
    board = Board()
    board.draw(window)
    moves = 0
    while not board.solved:
        # check for player quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        board.branch()

        color = board.next_move()
        board.change_color(color)
        moves += 1

        # print board
        board.draw(window)
    pygame.quit()

@numba.jit()
def neat_new_board(genomes, config):
    # Init NEAT
    nets = []
    players = []

    # pygame and window
    pygame.init()
    window = pygame.display.set_mode((600, 720))
    board = Board()
    board.draw(window)

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        # Init my cars
        players.append(Player(board))

    generation += 1
    all_solved = False

    while not all_solved:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        captures_before = [1]*len(players)
        all_solved = True
        # get next move
        for index, player in enumerate(players):
            if player.moves > 75:
                genomes[index][1].fitness -= 10000
                player.alive = False
            if not player.alive:
                continue
            captures_before[index] = player.board.count_captured()
            if player.is_solved():
                print("Solved!")
                player.alive = False
                genomes[index][1].fitness += (75 - player.moves)
                continue
            else:
                all_solved = False

            if all_solved:
                break

            # get neural input
            output = nets[index].activate(player.get_data())
            # get color choice from neural input (for max result)
            color_index = output.index(max(output))

            if list(Colors)[color_index].value == players[index].board.board[0][0].color:
                players[index].alive = False

            # play selected move
            player.board.change_color(list(Colors)[color_index].value)
            player.moves += 1



            # print board

        # print(
        #     "move: " + str(players[best_index].moves) +
        #     " - player: " + str(best_index) +
        #     " - captured: " + str(players[best_index].board.count_captured()) +
        #     " - fitness: " + str(genomes[best_index][1].fitness) +
        #     " - color_index: " + str(first_color_index) +
        #     " - color: " + str(list(Colors)[first_color_index])
        # )
        # players[best_index].draw_board(window)

        # update fitness of each players and
        for index, player in enumerate(players):
            genomes[index][1].fitness += (player.board.count_captured() - captures_before[index]) - 1

        pygame.time.delay(125)
        players[0].draw_board(window)

    # pygame.quit()

if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    #generation = 0
    #p.run(neat_new_board, 20)
    best_game = -1
    solve_new_board()




