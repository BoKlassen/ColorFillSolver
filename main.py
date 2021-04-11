import os
import pygame
import neat
import time
from Board import Board
from Player import Player
from Colors import Colors
from copy import deepcopy

global generation


def cap_window(window, method, suffix, board_size, width, depth, steps):
    rect = pygame.Rect(0, 0, window.get_rect().width, window.get_rect().height)
    sub = window.subsurface(rect)
    screenshot = pygame.Surface((window.get_rect().width, window.get_rect().height))
    screenshot.blit(sub, (0, 0))
    path = create_path_directory(method, board_size, width, depth, steps) + "/" + method + str(suffix) + ".jpg"
    pygame.image.save(screenshot, path)


def clear_extra_photos(prefix, start):
    print("removing photos...")
    while os.path.exists(prefix + "/" + prefix + str(start + 1) + ".jpg"):
        os.remove(prefix + "/" + prefix + str(start + 1) + ".jpg")
        start += 1
    print("Removing photos complete.")


def create_path_directory(method, board_size, width, depth, steps):
    directory = "games/" + method + "/" + method + "-" + str(board_size) + "-" + str(width) + "-" + str(depth) + "-" + str(steps)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


def smart_solve(board, window, draw, width, depth, steps):
    if depth < steps:
        print("Parameter steps must be less than depth")
    t = time.time()
    moves = 0

    while not board.solved:
        move_list = board.backtrack_util(depth, width)
        for i in range(steps):
            if draw:
                board.draw(window)
                cap_window(window, "smart", moves, board.size, width, depth, steps)
            board.change_color(move_list[i])
            moves += 1
            print("smart: " + str(moves))
            if board.solved:
                break
        if board.solved:
            break
    if draw:
        board.draw(window)
        cap_window(window, "smart", moves, board.size, width, depth, steps)
    res = (moves, time.time() - t)
    clear_extra_photos("smart", moves)
    return res


def neat_new_board(genomes, config):
    # Init NEAT
    nets = []
    players = []

    generation = 0

    # pygame and window
    pygame.init()
    window = pygame.display.set_mode((600, 600))
    board = Board(14)
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
            if player.moves > 50:
                genomes[index][1].fitness -= 2000
                player.alive = False
            if not player.alive:
                continue
            captures_before[index] = player.board.count_captured()
            if player.is_solved():
                print("Solved!")
                player.alive = False
                genomes[index][1].fitness += (50 - player.moves)
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
            player.board.draw(window)
            player.moves += 1

        # update fitness of each players and
        for index, player in enumerate(players):
            genomes[index][1].fitness += (player.board.count_captured() - captures_before[index]) - 1

        pygame.time.delay(125)
        players[0].draw_board(window)

    # pygame.quit()


if __name__ == "__main__":

    run_neat = False

    if run_neat:
        generation = 0
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
        p.run(neat_new_board, 40)
    else:
        board_size = 14
        #pygame
        pygame.init()
        window = pygame.display.set_mode((board_size*40 + 20, board_size*40 + 20))
        board_count = 1
        fast_times = 0
        smart_times = 0
        fast_moves = 0
        smart_moves = 0

        for _ in range(board_count):
            board = Board(board_size)
            fast_res = smart_solve(deepcopy(board), window, True, 6, 1, 1)
            fast_moves += fast_res[0]
            fast_times += fast_res[1]
            print("\nFast solve:\n" + str(board_count) + " solves in " + str(fast_moves) + " moves and " + str(fast_times) + " seconds.\nAverage: " + str(fast_times / board_count) + " seconds per solve.")
            fast_times = 0
            fast_moves = 0
            smart_res = smart_solve(deepcopy(board), window, True, 3, 6, 2)
            smart_moves += smart_res[0]
            smart_times += smart_res[1]
            print("\nSmart solve:\n" + str(board_count) + " solves in " + str(smart_moves) + " moves and " + str(smart_times) + " seconds.\nAverage: " + str(smart_times / board_count) + " seconds per solve.")

        print("\nFast solve:\n" + str(board_count) + " solves in " + str(fast_moves) + " moves and " + str(fast_times) + " seconds.\nAverage: " + str(fast_times / board_count) + " seconds per solve.")
        print("\nSmart solve:\n" + str(board_count) + " solves in " + str(smart_moves) + " moves and " + str(smart_times) + " seconds.\nAverage: " + str(smart_times / board_count) + " seconds per solve.")

    pygame.quit()



