# -*- coding: utf-8 -*-
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.merge import Add
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import np_utils
from collections import deque
import random
import numpy as np
import pickle

"""
Creat NN model
"""
class Residual_CNN():
    l2_const = 1e-4  # coef of l2 penalty 
    def __init__(self, input_dim, output_dim, hidden_layers = {"filters":128, "kernel_size":(3, 3)}, num_layers = 5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.num_layers = num_layers
        self.filters = hidden_layers["filters"]
        self.kernel_size = hidden_layers["kernel_size"]
        self.model = self._build_model()

    def conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first", padding = 'same', use_bias=False, activation='linear', kernel_regularizer = l2(self.l2_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return (x)

    def residual_layer_block(self, input_block, filters, kernel_size):
        x = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first", padding = 'same', use_bias=False, activation='linear', kernel_regularizer = l2(self.l2_const))(input_block)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first", padding = 'same', use_bias=False, activation='linear', kernel_regularizer = l2(self.l2_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        
        return (x)
    def value_head(self, x):
        x = Conv2D(filters = 1 , kernel_size = (1,1) , data_format="channels_first" , padding = 'same', use_bias=False, activation='linear', kernel_regularizer = l2(self.l2_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(20, use_bias=False, activation='linear', kernel_regularizer=l2(self.l2_const))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias=False, activation='tanh', kernel_regularizer=l2(self.l2_const), name = 'value_head')(x)
        return (x)

    def policy_head(self, x):
        x = Conv2D(filters = 2, kernel_size = (1,1), data_format="channels_first", padding = 'same', use_bias=False, activation='linear', kernel_regularizer = l2(self.l2_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(self.output_dim, use_bias=False, activation='softmax', kernel_regularizer=l2(self.l2_const), name = 'policy_head')(x)
        return (x)

    def _build_model(self):

        main_input = Input(self.input_dim, name = 'main_input')
        #Input
        x = self.conv_layer(main_input, self.filters, self.kernel_size)
        #hidden layers: Residual_CNN
        for i in range (self.num_layers):
            x = self.residual_layer_block(x, self.filters, self.kernel_size)

        # action policy layers
        self.policy_net = self.policy_head(x)
        # state value layers
        self.value_net = self.value_head(x)

        model = Model(inputs = [main_input], outputs = [self.policy_net, self.value_net])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
			optimizer=Adam(),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)
        
        return model

class PolicyValueNet(Residual_CNN):
    """policy-value network """
    buffer_size = 10000*2
    batch_size = 1024*2  # mini-batch size for training
    epochs = 5  # num of train_steps for each update
    data_buffer = deque(maxlen=buffer_size)
    kl_targ = 0.02
    learn_rate = 2e-3
    lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL

    def __init__(self, input_dim):
        self.input_dim = input_dim
        Residual_CNN.__init__(self, input_dim = self.input_dim, output_dim = self.input_dim[1]*self.input_dim[2])
        
    def propagation(self, input):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = input.availables
        current_state = input.current_state()

        current_state = current_state.reshape(-1, self.input_dim[0], self.input_dim[1], self.input_dim[1])
        current_state = np.array(current_state)
        act_probs, value = self.model.predict_on_batch(current_state)
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])

        return act_probs, value[0][0]

    def train(self, state_input, mcts_probs, winner, learning_rate):
        state_input_union = np.array(state_input)
        mcts_probs_union = np.array(mcts_probs)
        winner_union = np.array(winner)
        loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)

        K.set_value(self.model.optimizer.lr, learning_rate)
        self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
        return loss[0]
    
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.model.predict_on_batch(np.array(state_batch))
        for i in range(self.epochs):
            loss = self.train(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.model.predict_on_batch(np.array(state_batch))
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(("kl:{:.5f}," "lr_multiplier:{:.3f}," "loss:{}," ).format(kl, self.lr_multiplier, loss))
        return loss
    def memory(self, play_data):
        play_data = self.get_equi_data(list(play_data)[:])
        self.data_buffer.extend(play_data)

    def load_model(self, model_file):
        net_params = pickle.load(open(model_file, 'rb'))
        self.model.set_weights(net_params)
        print("Load paras from file: " + model_file)
    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.model.get_weights()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
        print("Save paras to file: " + model_file)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.input_dim[1], self.input_dim[2])), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

