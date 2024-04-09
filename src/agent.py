#!/usr/bin/python3
#  agent.py
#  Nine-Board Tic-Tac-Toe Agent starter code
#  COMP3411/9814 Artificial Intelligence
#  CSE, UNSW

import socket
import sys
import numpy as np
import math

# a board cell can hold:
#   0 - Empty
#   1 - We played here
#   2 - Opponent played here

EMPTY = 2

ILLEGAL_MOVE  = 0
STILL_PLAYING = 1
WIN           = 2
LOSS          = 3
DRAW          = 4

MAX_MOVE      = 9

MIN_EVAL = -1000000
MAX_EVAL =  1000000

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
s = [".","X","O"]
curr = 0 # this is the current board to play in

# print a row
def print_board_row(bd, a, b, c, i, j, k):
    print(" "+s[bd[a][i]]+" "+s[bd[a][j]]+" "+s[bd[a][k]]+" | " \
             +s[bd[b][i]]+" "+s[bd[b][j]]+" "+s[bd[b][k]]+" | " \
             +s[bd[c][i]]+" "+s[bd[c][j]]+" "+s[bd[c][k]])

# Print the entire board
def print_board(board):
    print_board_row(board, 1,2,3,1,2,3)
    print_board_row(board, 1,2,3,4,5,6)
    print_board_row(board, 1,2,3,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 4,5,6,1,2,3)
    print_board_row(board, 4,5,6,4,5,6)
    print_board_row(board, 4,5,6,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 7,8,9,1,2,3)
    print_board_row(board, 7,8,9,4,5,6)
    print_board_row(board, 7,8,9,7,8,9)
    print()

# Calling the Minimax algorithm
def doAlphaBeta(depth, turn):
    finalScore = []
    alphaBeta(depth, turn, math.inf, -math.inf)
    
def alphaBeta(player, m, board, alpha, beta, best_move):

    best_eval = MIN_EVAL

    if game_won( 1-player, board ):   # LOSS
        return -1000 + m  # better to win faster (or lose slower)

    this_move = 0
    for r in range( 1, 10):
        if board[r] == EMPTY:         # move is legal
            this_move = r
            board[this_move] = player # make move
            this_eval = -alphaBeta(1-player,m+1,board,-beta,-alpha,best_move)
            board[this_move] = EMPTY  # undo move
            if this_eval > best_eval:
                best_move[m] = this_move
                best_eval = this_eval
                if best_eval > alpha:
                    alpha = best_eval
                    if alpha >= beta: # cutoff
                        return( alpha )

    if this_move == 0:  # no legal moves
        return( 0 )     # DRAW
    else:
        return( alpha )

# choose a move to play
def play():
    # print_board(boards)

    # just play a random move for now
    n = np.random.randint(1,9)
    while boards[curr][n] != 0:
        n = np.random.randint(1,9)

    # print("playing", n)
    place(curr, n, 1)
    return n

# place a move in the global boards
def place( board, num, player ):
    global curr
    curr = num
    boards[board][num] = player
    
# Check if game is won
def game_won(p, bd):
    return(  ( bd[1][0] == p and bd[2][0] == p and bd[3][0] == p )
           or( bd[4][0] == p and bd[5][0] == p and bd[6][0] == p )
           or( bd[7][0] == p and bd[8][0] == p and bd[9][0] == p )
           or( bd[1][0] == p and bd[4][0] == p and bd[7][0] == p )
           or( bd[2][0] == p and bd[5][0] == p and bd[8][0] == p )
           or( bd[3][0] == p and bd[6][0] == p and bd[9][0] == p )
           or( bd[1][0] == p and bd[5][0] == p and bd[9][0] == p )
           or( bd[3][0] == p and bd[5][0] == p and bd[7][0] == p ))
    
# check if one cell is won
def check_minigame_won( p, bd, curr ):
    if ( bd[curr][1] == p and bd[curr][2] == p and bd[curr][3] == p )\
        or ( bd[curr][4] == p and bd[curr][5] == p and bd[curr][6] == p )\
        or( bd[curr][7] == p and bd[curr][8] == p and bd[curr][9] == p )\
        or( bd[curr][1] == p and bd[curr][4] == p and bd[curr][7] == p )\
        or( bd[curr][2] == p and bd[curr][5] == p and bd[curr][8] == p )\
        or( bd[curr][3] == p and bd[curr][6] == p and bd[curr][9] == p )\
        or( bd[curr][1] == p and bd[curr][5] == p and bd[curr][9] == p )\
        or( bd[curr][3] == p and bd[curr][5] == p and bd[curr][7] == p ):
        bd[curr][0] = p

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
        place(int(args[0]), int(args[1]), 2)
        return play()  # choose and return the second move

    # third_move(K,L,M) means that the first and second move were
    # in square L of sub-board K, and square M of sub-board L,
    # and we are expected to return the third move.
    elif command == "third_move":
        # place the first move (randomly generated for us)
        place(int(args[0]), int(args[1]), 1)
        # place the second move (chosen by opponent)
        place(curr, int(args[2]), 2)
        return play() # choose and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        return play() # choose and return our next move

    elif command == "win":
        print("Yay!! We win!! :)")
        return -1

    elif command == "loss":
        print("We lost :(")
        return -1

    return 0

# connect to socket
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(sys.argv[2]) # Usage: ./agent.py -p (port)

    s.connect(('localhost', port))
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
