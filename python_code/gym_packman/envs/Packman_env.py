#!/usr/bin/env python
# coding: utf-8

# # 1. Build Packman Environment with OpenAI Gym

import random
from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from IPython import display

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

class PackmanEnv(Env):

    metadata = {'render.modes': ['human']}
    
    rewards = {
        'Start': 0.5, #0.5,
        0: -0.001, #stay
        1: -0.005, #left
        2: -0.005, #up
        3: -0.005, #right
        4: -0.005, #down
        'CollectDirt': 0,  # (-2 + 2 = 0)
        'EndGame': 1.0,
        'invalidAction': -0.005
    }

    toIndex = {
        'Board': 0,
        'Human trace': 1,
        'Computer trace': 2,
        'Human awards': 3,
        'Computer awards': 4,
        'All awards': 5,
    }

    def __init__(self):
        super(PackmanEnv, self).__init__()
        # Actions we can take, left, down, stay, up, right
        self.action_space = Discrete(5)

        # Define a 2-D observation space
        # self.observation_shape = (10, 10, 6)
        self.observation_shape = (10, 10, 3)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                     high=np.ones(self.observation_shape),
                                     dtype=np.float32)

        # Set start state
        self.canvas = None
        self.state = None

        # Load human model from the computer
        self.human_model = tf.keras.models.load_model('./data/humanModel/model_v0.h5')
        # self.human_model.summary()

    def step(self, action):
        # Apply action
        # 0: "Stay"
        # 1: "Left"
        # 2: "Up"
        # 3: "Right"
        # 4: "Down"
        computer_reward = 0
        human_reward = 0

        self.step_num += 1
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action: " + str(action)

        human_pos = np.where(self.state[::, ::, self.toIndex['Human trace']] == 1)
        computer_pos = np.where(self.state[::, ::, self.toIndex['Computer trace']] == 1)

        # predict next human action

        # when human model is ready uncomment this line
        human_action = self.predict_action(self.canvas)
        # if np.random.random() < 1.0:
        #     action = self.get_random_valid_action('computer')
        # else:
        #     action = 0

        # assert self.valid_action(action, 'computer'), "Computer preformed invalid action: " + str(
        #     action) + " at pos: " + str(computer_pos)
        # assert self.valid_action(human_action, 'human'), "Human preformed invalid action: " + str(
        #     human_action) + " at pos: " + str(computer_pos)

        computer_valid_action = self.valid_action(action, 'computer')
        human_valid_action = self.valid_action(human_action, 'human')
        if not computer_valid_action:
            computer_reward += self.rewards['invalidAction']
            action = 0
        elif not human_valid_action:
            human_reward += self.rewards['invalidAction']
            human_action = 0
       
        # computer and human action are valid
        self.move(human_action, 'human')
        self.move(action, 'computer')

        # check for clean dirt for both agents
        dirts_pos = np.where(self.state[::, ::, self.toIndex['All awards']] == 1)
        for dirt_pos_i, dirt_pos_j in zip(dirts_pos[0], dirts_pos[1]):
            if human_pos[0][0] == dirt_pos_i and human_pos[1][0] == dirt_pos_j:
                self.state[::, ::, self.toIndex['All awards']][human_pos] = 0
                self.state[::, ::, self.toIndex['Human awards']][human_pos] = 1
                human_reward += self.rewards['CollectDirt']

            if computer_pos[0][0] == dirt_pos_i and computer_pos[1][0] == dirt_pos_j:
                self.state[::, ::, self.toIndex['All awards']][computer_pos] = 0
                self.state[::, ::, self.toIndex['Computer awards']][computer_pos] = 1
                computer_reward += self.rewards['CollectDirt']

        # Reward for executing an action.
        computer_reward += self.rewards[action]
        human_reward += self.rewards[human_action]

        if not np.any(self.state[::, ::, self.toIndex['All awards']]):  # game ended when there is no dirt to clean
            computer_reward += self.rewards['EndGame']
            human_reward += self.rewards['EndGame']
            done = True
        else:
            # Flag that marks the termination of an episode
            done = False


        self.ep_return += computer_reward
        self.ep_human_reward += human_reward

        # Set placeholder for info
        info = {
            'done': done,
            'current_reward': computer_reward,
            'human_reward': human_reward,
            'computer valid action': computer_valid_action,
            # 'computer action': action,
            'human valid action': human_valid_action
            # 'human action': human_action
        }

        # Return step information
        # return self.state, computer_reward, done, info
        return self.render(), computer_reward, done, info

    def render(self, mode='rgb_array'):
        screen_width = 400
        screen_height = 400
        # Implement viz
        r = self.state[::, ::, self.toIndex['Human awards']] / 2 + self.state[::, ::, self.toIndex['Human trace']] + self.state[::, ::, self.toIndex['All awards']]
        g = self.state[::, ::, self.toIndex['Board']] / 3 + self.state[::, ::, self.toIndex['All awards']]
        b = self.state[::, ::, self.toIndex['Computer awards']] / 2 + self.state[::, ::, self.toIndex['Computer trace']] + self.state[::, ::, self.toIndex['All awards']]
        rgb = np.dstack((b, g, r))
        self.canvas = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))  # NormalizeImage

        # Render the environment to the screen
        if mode == 'human':
            name = self.ep_human_reward
            cv2.namedWindow('Packman-v0', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Packman-v0', screen_width, screen_height)
            # canvas_bgr = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
            cv2.imshow('Packman-v0',self.canvas)
            cv2.waitKey(1)

        elif mode == 'jupyter':
            ## Display for jupyter notebook
            if self.call_once:
                self.call_once = False
                plt.figure(figsize=(4, 4))
                self.img = plt.imshow(self.canvas)  # only call this once

            self.img.set_data(self.canvas)  # just update the data
            plt.axis('off')
            info = {
                'ep_return': self.ep_return,
                'human_return': self.ep_human_reward,
            }
            plt.title(f'info: {info}')
            display.display(plt.gcf())
            display.clear_output(wait=True)

        elif mode == 'rgb_array':
            return self.canvas

    def reset(self):
        self.call_once = True
        self.step_num = 0
        # Reset board game
        self.state = self.init_state()
        # Reset the reward
        self.ep_return = self.rewards['Start']
        self.ep_human_reward = self.rewards['Start']

        # return self.state
        return self.render(mode='rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # np.random.seed(seed)
        # tf.random.set_seed(seed)
        # random.seed(seed)
        return [seed]

    def close(self):
        # if self.canvas != None:
        cv2.destroyAllWindows()
            
    #################### Helper functions ########################

    def init_state(self):
        # init board state with random n=5 dirt position
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        human_trace = np.zeros(board.shape)
        human_trace[5][3] = 1  # locate human player
        board[5][3] = 0  # locate human player

        computer_trace = np.zeros(board.shape)
        computer_trace[6][4] = 1  # locate computer player
        board[6][4] = 0  # locate computer player

        human_awards = np.zeros(board.shape)
        computer_awards = np.zeros(board.shape)

        all_awards = np.zeros(board.shape)
        # idx = np.random.choice(np.count_nonzero(board), 5)
        
        # Determine fixed dirt locations
        idx = [ 10, 23, 30, 35, 41]
        all_awards[tuple(map(lambda x: x[idx], np.where(board)))] = 1

        board[5][3] = 1  # locate human player
        board[6][4] = 1  # locate computer player

        return np.concatenate([np.expand_dims(board, axis=2), np.expand_dims(human_trace, axis=2),
         np.expand_dims(computer_trace, axis=2), np.expand_dims(human_awards, axis=2),
         np.expand_dims(computer_awards, axis=2), np.expand_dims(all_awards, axis=2)], axis=2)
        
    def get_random_valid_action(self, who):
        random_action = random.randrange(self.action_space.n) #self.action_space.sample()
        while not self.valid_action(random_action, who):
            random_action = random.randrange(self.action_space.n) #self.action_space.sample()
        return random_action

    def valid_action(self, action, who):
        if who == 'human':
            pos = np.where(self.state[::, ::, self.toIndex['Human trace']] == 1)
        else:
            pos = np.where(self.state[::, ::, self.toIndex['Computer trace']] == 1)
        next_pos = self.new_pos(pos, action)
        if self.state[next_pos[0], next_pos[1], self.toIndex['Board']] == 0:
            return False
        else:
            return True

    def new_pos(self, current_pos, action):
        if action == 0:  # stay
            return current_pos
        elif action == 1:  # left
            return (current_pos[0], current_pos[1] - 1)
        elif action == 2:  # up
            return (current_pos[0] - 1, current_pos[1])
        elif action == 3:  # right
            return (current_pos[0], current_pos[1] + 1)
        elif action == 4:  # down
            return (current_pos[0] + 1, current_pos[1])
        else:  # default case if action is not found
            assert True, "action: " + str(action) + " is not a valid action at agent pos: " + str(current_pos)

    def move(self, action, agent):
        # assume action is valid
        if agent == 'human':
            current_pos = np.where(self.state[::, ::, self.toIndex['Human trace']] == 1)
            self.state[::, ::, self.toIndex['Human trace']] = self.state[::, ::, self.toIndex['Human trace']] * 0.9
            next_pos = self.new_pos(current_pos, action)
            self.state[next_pos[0], next_pos[1], self.toIndex['Human trace']] = 1

        elif agent == 'computer':
            current_pos = np.where(self.state[::, ::, self.toIndex['Computer trace']] == 1)
            self.state[::, ::, self.toIndex['Computer trace']] = self.state[::, ::, self.toIndex['Computer trace']] * 0.9
            next_pos = self.new_pos(current_pos, action)
            self.state[next_pos[0], next_pos[1], self.toIndex['Computer trace']] = 1
        else:
            assert True, "agent not define:" + str(agent)

    def predict_action(self, img):
        b, g, r = cv2.split(img) # For BGR image
        img = np.dstack((r, g, b))
        # img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img, 0)  # Create a batch
        predictions = self.human_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        action = np.argmax(score)
        # print(score)
        # print(action)
        #         print(
        #             "This image most likely belongs to {} with a {:.2f} percent confidence."
        #             .format(action, 100 * np.max(score))
        #         )
        while(not self.valid_action(action, 'human')):
            score = np.delete(score, action)
            action = np.argmax(score)

        return action