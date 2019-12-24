# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game():
    """game server"""
    game_name = "Gobang"
    board_width = 10
    board_height = 10
    n_in_row = 5
    flag_human_click = False
    move_human = -1
    piece_size = 0.4
    margin = 0.5
    def __init__(self, flag_is_shown = True, flag_is_train = True):
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        fig = plt.figure(self.game_name)
        plt.ion()   #enable interaction
        self.ax = plt.subplot(111)
        self.canvas = fig.canvas
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.set_bg()
    def onclick(self, event):
        x, y = event.xdata, event.ydata
        try:
            x = int(np.round(x))
            y = int(np.round(y))
        except TypeError:
            print("Invalid area")
        else:
            self.move_human = x*self.board_width + y
            self.flag_human_click = True
    def set_bg(self):
        plt.cla()
        plt.title(self.game_name)
        plt.grid(linestyle='-')
        plt.axis([-self.margin, self.board.width - self.margin, - self.margin, self.board.height - self.margin])
        x_major_locator=MultipleLocator(1)
        y_major_locator=MultipleLocator(1)
        self.ax.xaxis.set_major_locator(x_major_locator)
        self.ax.yaxis.set_major_locator(y_major_locator)
    def graphic(self, board):
        """Draw the board and show game info"""
        if board.current_player == 1:
            color='blue'
        elif board.current_player == 2:
            color='black'
        x = board.last_move // board.width
        y = board.last_move % board.height
        plt.text(x, y, ("%d" % len(board.states)), c = "white")
        self.ax.add_artist(plt.Circle((x, y), self.piece_size, color= color))   #repulsion_circle
        plt.pause(0.001)
    def start_self_play(self, player):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, self.flag_is_train)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if self.flag_is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if self.flag_is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                self.set_bg()
                return winner, zip(states, mcts_probs, winners_z)
    def start_play(self, player):
        """ start a game using a MCTS player
        """
        self.board.init_board(0)
        while True:
            if (self.board.current_player == 1):
                move, move_probs = player.get_action(self.board, self.flag_is_train)
                # perform a move
                self.board.do_move(move)
            else:
                if(self.flag_human_click):
                    if self.move_human in self.board.availables:
                        self.flag_human_click = False
                        self.board.do_move(self.move_human)
                    else:
                        self.flag_human_click = False
                        print("Invalid input")
            if self.flag_is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                # reset MCTS root node
                player.reset_player()
                if self.flag_is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                self.set_bg()
                break
            
