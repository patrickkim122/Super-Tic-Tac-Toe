#!/usr/bin/python3
#  agent.py
#  Nine-Board Tic-Tac-Toe Agent starter code
#  COMP3411/9814 Artificial Intelligence
#  CSE, UNSW

# SUMMARY:
# This Python script employs a meticulously designed algorithmic structure to
# guide an AI agent to victory in Nine-Board Tic-Tac-Toe. At its core, the
# system features dynamic assessment functions for individual mini-boards and
# the overall game board, calculating scores based on critical game situations
# and strategic placements. Leveraging numpy arrays for robust representation,
# the agent adeptly manages the game's complexity. Decision-making is powered by
# the minimax algorithm, augmented with alpha-beta pruning for computational
# efficiency. This algorithm navigates potential game states recursively,
# evaluating board states and selecting optimal moves based on calculated
# scores. The AI agent adeptly responds to opponent actions while strategically
# positioning itself for victory in the dynamic landscape of Nine-Board
# Tic-Tac-Toe.

# To enhance the AI agent's performance in Nine-Board Tic-Tac-Toe, several
# design choices were carefully made. The utilisation of NumPy arrays for game
# state representation ensured efficient board manipulation, striking a balance
# between simplicity and computational speed. Separating evaluation functions
# for mini-boards and the overall game board enabled nuanced assessments of game
# states, considering factors like winning positions and strategic advantages.
# The decision to employ the minimax algorithm with alpha-beta pruning aimed to
# optimise computational resources while ensuring optimal move selection.
# Additionally, incorporating input parsing facilitated dynamic adaptation to
# opponent moves, enabling informed decision-making and strategic adjustments.
# Finally, imposing a depth limitation on the minimax search tree maintained a
# balance between decision quality and computational efficiency, ensuring smooth
# gameplay while maximising the AI agent's performance in navigating the
# complexities of Nine-Board Tic-Tac-Toe.

import socket
import sys
import numpy as np

EMPTY = 0

ILLEGAL_MOVE = 0
STILL_PLAYING = 1
WIN = 2
LOSS = 3
DRAW = 4

MAX_MOVE = 9

MIN_EVAL = -1000000
MAX_EVAL = 1000000

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
s = [".", "X", "O"]
curr = 0  # this is the current board to play in
max_depth = 0


# print a row
def print_board_row(bd, a, b, c, i, j, k):
    print(
        " "
        + s[bd[a][i]]
        + " "
        + s[bd[a][j]]
        + " "
        + s[bd[a][k]]
        + " | "
        + s[bd[b][i]]
        + " "
        + s[bd[b][j]]
        + " "
        + s[bd[b][k]]
        + " | "
        + s[bd[c][i]]
        + " "
        + s[bd[c][j]]
        + " "
        + s[bd[c][k]]
    )


# Print the entire board
def print_board(board):
    print_board_row(board, 1, 2, 3, 1, 2, 3)
    print_board_row(board, 1, 2, 3, 4, 5, 6)
    print_board_row(board, 1, 2, 3, 7, 8, 9)
    print(" ------+-------+------")
    print_board_row(board, 4, 5, 6, 1, 2, 3)
    print_board_row(board, 4, 5, 6, 4, 5, 6)
    print_board_row(board, 4, 5, 6, 7, 8, 9)
    print(" ------+-------+------")
    print_board_row(board, 7, 8, 9, 1, 2, 3)
    print_board_row(board, 7, 8, 9, 4, 5, 6)
    print_board_row(board, 7, 8, 9, 7, 8, 9)
    print()


def possible_moves(board, miniboard):
    moves = []
    for i in range(1, 10):
        if board[miniboard][i] == EMPTY:
            moves.append(i)
    return moves


# Evaluates a single miniboard
def evaluate_miniboard(miniboard, player, move_hist):
    sum = 0
    multiplier = 1
    # Set a multiplier
    # If miniboard is going to be played next, double the number
    if miniboard == move_hist[-1]:
        multiplier = 2
    # Check all three horizontal lines, all three vertical lines and both diagonal lines
    for indices in (
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
        [1, 5, 9],
        [3, 5, 7],
    ):
        line = []
        for index in indices:
            line.append(boards[miniboard][index])
            # if there are 3 in a row
            if line.count(player) == 3:
                sum += 1000 * multiplier
            # 2 in a row and last one is blank
            elif line.count(player) == 2 and line.count(3 - player) == 0:
                sum += 100 * multiplier
            # other player
            elif line.count(3 - player) == 3:
                sum -= 1000 * multiplier
            elif line.count(3 - player) == 2 and line.count(player) == 0:
                sum -= 100 * multiplier
    return sum + np.random.rand()


# This functions evaluates the whole board
def evaluate_board(depth, move_hist):
    sum = 0
    if depth != 0:
        if game_won(1, boards, move_hist[-1]):
            sum += 10000
        else:
            sum -= 10000
        return sum
    for miniboard in range(1, 10):
        sum += evaluate_miniboard(miniboard, 1, move_hist)
    return sum + 100 * depth


# AI turn
# This function will call the minimax function
# After all the calculations are done, the move with the highest score is used
# You need to figure out how to give each move a score in evaluate_board
def next_move(player, depth):
    best_score = float("-inf")
    best_move = None
    for move in possible_moves(boards, curr):
        move_hist = []
        this_move = move
        boards[curr][this_move] = player
        move_hist.append(this_move)
        score = minimax(
            depth, float("-inf"), float("inf"), False, move_hist
        )  # Depth is set to 5 for speed
        boards[curr][this_move] = EMPTY
        move_hist.pop()
        if score > best_score:
            best_score = score
            best_move = this_move
    return best_move


# Minimax Recursive Algorithm
def minimax(depth, alpha, beta, is_maximising, move_hist):
    if (
        depth == 0
        or game_won(1, boards, move_hist[-1])
        or game_won(2, boards, move_hist[-1])
    ):
        return evaluate_board(depth, move_hist)
    if is_maximising:
        max_eval = float("-inf")
        for move in possible_moves(boards, move_hist[-1]):
            boards[move_hist[-1]][move] = 1
            move_hist.append(move)
            eval = minimax(depth - 1, alpha, beta, False, move_hist)
            move_hist.pop()
            boards[move_hist[-1]][move] = EMPTY
            max_eval = max(max_eval, eval)
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in possible_moves(boards, move_hist[-1]):
            boards[move_hist[-1]][move] = 2
            move_hist.append(move)
            eval = minimax(depth - 1, alpha, beta, True, move_hist)
            move_hist.pop()
            boards[move_hist[-1]][move] = EMPTY
            min_eval = min(min_eval, eval)
            beta = min(beta, min_eval)
            if beta <= alpha:
                break
        return min_eval


# choose a move to play
def play():
    # Calculate the next move
    # Currently max depth is hard set at 5
    global max_depth
    max_depth = 3
    n = next_move(1, max_depth)
    place(curr, n, 1)
    return n


# place a move in the global boards
def place(board, num, player):
    global curr
    curr = num
    boards[board][num] = player


# check if one cell is won
def game_won(p, bd, curr):
    return (
        (bd[curr][1] == p and bd[curr][2] == p and bd[curr][3] == p)
        or (bd[curr][4] == p and bd[curr][5] == p and bd[curr][6] == p)
        or (bd[curr][7] == p and bd[curr][8] == p and bd[curr][9] == p)
        or (bd[curr][1] == p and bd[curr][4] == p and bd[curr][7] == p)
        or (bd[curr][2] == p and bd[curr][5] == p and bd[curr][8] == p)
        or (bd[curr][3] == p and bd[curr][6] == p and bd[curr][9] == p)
        or (bd[curr][1] == p and bd[curr][5] == p and bd[curr][9] == p)
        or (bd[curr][3] == p and bd[curr][5] == p and bd[curr][7] == p)
    )


# read what the server sent us and
# parse only the strings that are necessary
def parse(string):
    if "(" in string:
        command, args = string.split("(")
        args = args.split(")")[0]
        args = args.split(",")
    else:
        command, args = string, []

    # init tells us that a new game is about to begin.
    # start(x) or start(o) tell us whether we will be playing first (x)
    # or second (o); we might be able to ignore start if we internally
    # use 'X' for *our* moves and 'O' for *opponent* moves.

    # second_move(K,L) means that the (randomly generated)
    # first move was into square L of sub-board K,
    # and we are expected to return the second move.
    if command == "second_move":
        # place the first move (randomly generated for opponent)
        print(f"first move was {int(args[0]), int(args[1])}")
        place(int(args[0]), int(args[1]), 2)
        return play()  # choose and return the second move

    # third_move(K,L,M) means that the first and second move were
    # in square L of sub-board K, and square M of sub-board L,
    # and we are expected to return the third move.
    elif command == "third_move":
        # place the first move (randomly generated for us)
        place(int(args[0]), int(args[1]), 1)
        print(f"first move was {int(args[0]), int(args[1])}")
        # place the second move (chosen by opponent)
        print(f"second move was {curr, int(args[2])}")
        place(curr, int(args[2]), 2)
        return play()  # choose and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        return play()  # choose and return our next move

    elif command == "win":
        print("Yay!! We win!! :)")
        return -1

    elif command == "loss":
        print("We lost :(")
        return -1

    return 0


# Connect to socket
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(sys.argv[2])  # Usage: ./agent.py -p (port)

    s.connect(("localhost", port))
    while True:
        text = s.recv(1024).decode()
        if not text:
            continue
        for line in text.split("\n"):
            response = parse(line)
            if response == -1:
                s.close()
                return
            elif response > 0:
                s.sendall((str(response) + "\n").encode())


if __name__ == "__main__":
    main()
