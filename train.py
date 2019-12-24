# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from env import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_keras import PolicyValueNet # Keras

class TrainPipeline():
    save_ParaFreq = 200
    MAX_EPISODES = 2000
    def __init__(self, flag_is_shown = False, flag_is_train = True):
        # training params
        self.flag_is_shown = flag_is_shown
        self.flag_is_train = flag_is_train
        self.game = Game(self.flag_is_shown, self.flag_is_train)
        self.NN = PolicyValueNet((4, self.game.board_width, self.game.board_height))
        if not self.flag_is_train:
            self.NN.load_model("./paras/policy.model")
        self.mcts_player = MCTSPlayer(self.NN.propagation)

    def train(self):
        """run the training pipeline"""
        for episode in range(self.MAX_EPISODES):
            if self.flag_is_train:
                winner, play_data = self.game.start_self_play(self.mcts_player)
                self.NN.memory(play_data)
                if len(self.NN.data_buffer) > self.NN.batch_size:
                    loss = self.NN.policy_update()
                else:
                    print("Collecting data: %d%%, " % (len(self.NN.data_buffer)/self.NN.batch_size*100), end="")
                # and save the model params
                if (episode+1) % self.save_ParaFreq == 0:
                    self.NN.save_model('./paras/policy.model')
                print("episode = %d" % episode)
            else:
                self.game.start_play(self.mcts_player)

if __name__ == '__main__':
    training_pipeline = TrainPipeline(flag_is_shown = True, flag_is_train = False)
    training_pipeline.train()
