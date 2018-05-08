import argparse
import numpy as np
import copy
import time

parser = argparse.ArgumentParser()


def move(Old_Board, action):

    # Create a copy of the board so that the old board doesn't change.
    Board = copy.copy(Old_Board)

    # Redefine the action to be 1 less, since python starts arrays at 0
    action -= 1

    # Figure out what row to place the piece in
    if sum(Board[:, action]) == 0:
        new_row = 5
    else:
        new_row = next(
                (i for i, x in enumerate(Board[:, action]) if x), None
                ) - 1
    
    # Figure out whose turn it is
    if sum(sum(Board == 1)) == sum(sum(Board == 2)):
        player_turn = 1
    else:
        player_turn = 2

    # Place that player's piece where they wanted it to go.
    Board[new_row, action] = player_turn

    # Initialize checks for 4-in-a-row's
    run_total = 0
    max_run = 0

    # Horizontals
    if 3 <= action <= 6:
        action_list = range(action-3, action+1)
        run_total = sum(Board[new_row, action_list] == player_turn)
        if run_total > max_run:
            max_run = run_total

    if 2 <= action <= 5:
        action_list = range(action-2, action+2)
        run_total = sum(Board[new_row, action_list] == player_turn)
        if run_total > max_run:
            max_run = run_total

    if 1 <= action <= 4:
        action_list = range(action-1, action+3)
        run_total = sum(Board[new_row, action_list] == player_turn)
        if run_total > max_run:
            max_run = run_total

    if 0 <= action <= 3:
        action_list = range(action, action+4)
        run_total = sum(Board[new_row, action_list] == player_turn)
        if run_total > max_run:
            max_run = run_total

    # Vertical
    if 0 <= new_row <= 2:
        action_list = range(new_row, new_row+4)
        run_total = sum(Board[action_list, action] == player_turn)
        if run_total > max_run:
            max_run = run_total

    # Diagonals
    Board_list = np.array([
                    [Board[0, 0], Board[1, 1], Board[2, 2], Board[3, 3]],
                    [Board[1, 0], Board[2, 1], Board[3, 2], Board[4, 3]],
                    [Board[2, 0], Board[3, 1], Board[4, 2], Board[5, 3]],
                    [Board[0, 1], Board[1, 2], Board[2, 3], Board[3, 4]],
                    [Board[1, 1], Board[2, 2], Board[3, 3], Board[4, 4]],
                    [Board[2, 1], Board[3, 2], Board[4, 3], Board[5, 4]],
                    [Board[0, 2], Board[1, 3], Board[2, 4], Board[3, 5]],
                    [Board[1, 2], Board[2, 3], Board[3, 4], Board[4, 5]],
                    [Board[2, 2], Board[3, 3], Board[4, 4], Board[5, 5]],
                    [Board[0, 3], Board[1, 4], Board[2, 5], Board[3, 6]],
                    [Board[1, 3], Board[2, 4], Board[3, 5], Board[4, 6]],
                    [Board[2, 3], Board[3, 4], Board[4, 5], Board[5, 6]],
                    [Board[0, 3], Board[1, 2], Board[2, 1], Board[3, 0]],
                    [Board[1, 3], Board[2, 2], Board[3, 1], Board[4, 0]],
                    [Board[2, 3], Board[3, 2], Board[4, 1], Board[5, 0]],
                    [Board[0, 4], Board[1, 3], Board[2, 2], Board[3, 1]],
                    [Board[1, 4], Board[2, 3], Board[3, 2], Board[4, 1]],
                    [Board[2, 4], Board[3, 3], Board[4, 2], Board[5, 1]],
                    [Board[0, 5], Board[1, 4], Board[2, 3], Board[3, 2]],
                    [Board[1, 5], Board[2, 4], Board[3, 3], Board[4, 2]],
                    [Board[2, 5], Board[3, 4], Board[4, 3], Board[5, 2]],
                    [Board[0, 6], Board[1, 5], Board[2, 4], Board[3, 3]],
                    [Board[1, 6], Board[2, 5], Board[3, 4], Board[4, 3]],
                    [Board[2, 6], Board[3, 5], Board[4, 4], Board[5, 3]],
                    ])
    run_total = max(sum(
            Board_list.transpose() == player_turn * np.ones((4, 24), dtype=int)
            ))
    if run_total > max_run:
        max_run = run_total

    if max_run >= 4:
        Reward = 1
    elif sum(sum(Board == 0)) == 0:
        Reward = -1
    else:
        Reward = 0

    return Board, Reward
