#!/usr/bin/env python
# coding: utf-8

# # 1. Build Packman Environment with OpenAI Gym

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import tensorflow as tf

class PackmanEnv(Env):

    metadata = {'render.modes': ['human']}
    
    rewards = {
        'Start': 50,
        0: -1,
        1: -2,
        2: -2,
        3: -2,
        4: -2,
        'CollectDirt': 2,  # (-2 + 2 = 0)
        'EndGame': 100
    }
    actions = {
        0: "Stay",
        1: "Left",
        2: "Up",
        3: "Right",
        4: "Down"
    }

    def __init__(self):
        super(PackmanEnv, self).__init__()
        # Actions we can take, left, down, stay, up, right
        self.action_space = Discrete(5)

        # Define a 2-D observation space
        self.observation_shape = (10, 10, 3)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                     high=np.ones(self.observation_shape),
                                     dtype=np.float32)
        self.seed()
        self.viewer = None

        # Set start state
        self.dict_state = None
        self.canvas = None

        # Load human model from the computer
        self.human_model = tf.keras.models.load_model('./data/humanModel/mode_v0')

    def step(self, action):
        # Apply action
        # 0: "Stay"
        # 1: "Left"
        # 2: "Up"
        # 3: "Right"
        # 4: "Down"
        computer_reward = 0
        human_reward = 0

        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        human_pos = np.where(self.dict_state['Human trace'] == 1)
        computer_pos = np.where(self.dict_state['Computer trace'] == 1)
        # predict next human action

        # when human model is ready uncomment this line
        #         human_action = self.predict_action(self.canvas)
        human_action = self.get_random_valid_action('human')

        assert self.valid_action(action, 'computer'), "Computer preformed invalid action: " + str(
            action) + " at pos: " + str(computer_pos)
        assert self.valid_action(human_action, 'human'), "Human preformed invalid action: " + str(
            human_action) + " at pos: " + str(computer_pos)

        self.move(human_action, 'human')  # assume human action is valid
        # apply the action to the agent
        self.move(action, 'computer')

        # check for clean dirt for both agents
        dirts_pos = np.where(self.dict_state['All awards'] == 1)
        for dirt_pos_i, dirt_pos_j in zip(dirts_pos[0], dirts_pos[1]):
            if human_pos[0][0] == dirt_pos_i and human_pos[1][0] == dirt_pos_j:
                self.dict_state['All awards'][human_pos] = 0
                self.dict_state['Human awards'][human_pos] = 1
                human_reward += self.rewards['CollectDirt']
            if computer_pos[0][0] == dirt_pos_i and computer_pos[1][0] == dirt_pos_j:
                self.dict_state['All awards'][computer_pos] = 0
                self.dict_state['Computer awards'][computer_pos] = 1
                computer_reward += self.rewards['CollectDirt']

        # Reward for executing an action.
        computer_reward = self.rewards[action]
        human_reward = self.rewards[human_action]

        if not np.any(self.dict_state['All awards']):  # game ended when there is no dirt to clean
            computer_reward += self.rewards['EndGame']
            human_reward += self.rewards['EndGame']
            done = True
        else:
            # Flag that marks the termination of an episode
            done = False

        self.ep_return += computer_reward
        self.ep_human_reward += human_reward

        self.canvas = self.convertToImage(self.dict_state)

        # Set placeholder for info
        info = {
            'done': done,
            'current_reward': computer_reward,
            'ep_return': self.ep_return,
            'human_return': self.ep_human_reward,
        }

        # Return step information
        return self.canvas, computer_reward, done, info

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        tileW = 40
        tileH = 40

        # Implement viz
        self.canvas = self.convertToImage(self.dict_state)

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            # self.viewer = rendering.SimpleImageViewer(maxwidth=screen_width)
            # self.viewer.imshow(self.canvas[:, :, ::-1])

            board = self.dict_state['Board']
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board[i][j] == 0:
                        tile = rendering.FilledPolygon([(i*tileW, i*tileH), (j*tileW, j*tileH), (tileW, tileW), (tileH, tileH)])
                        # ctx.fillRect( x*tileW, y*tileH, tileW, tileH);
                        self.viewer.add_geom(tile)
                    else:
                        tile = rendering.FilledPolygon([(i*tileW, i*tileH), (j*tileW, j*tileH), (tileW, tileW), (tileH, tileH)])
                        tile.set_color(0.8, 0.6, 0.4)
                        self.viewer.add_geom(tile)
            

        if self.canvas is None:
            return None
        
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
        # # Render the environment to the screen
        # if mode == 'human':
        #     if self.call_once:
        #         self.call_once = False
        #         plt.figure(figsize=(8, 8))
        #         self.img = plt.imshow(self.canvas)  # only call this once

        #     self.img.set_data(self.canvas)  # just update the data
        #     plt.axis('off')
        #     info = {
        #         'ep_return': self.ep_return,
        #         'human_return': self.ep_human_reward,
        #     }
        #     plt.title(f'info: {info}')
        #     display.display(plt.gcf())
        #     display.clear_output(wait=True)
        # elif mode == 'rgb_array':
        #     return self.canvas

    def reset(self):
        self.call_once = True
        # Reset board game
        self.dict_state = self.init_state()
        self.canvas = self.convertToImage(self.dict_state)
        # Reset the reward
        self.ep_return = self.rewards['Start']
        self.ep_human_reward = self.rewards['Start']

        return self.canvas

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
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
        human_trace[2][2] = 1  # locate human player
        board[2][2] = 0  # locate human player

        computer_trace = np.zeros(board.shape)
        computer_trace[7][7] = 1  # locate computer player
        board[7][7] = 0  # locate computer player

        human_awards = np.zeros(board.shape)
        computer_awards = np.zeros(board.shape)

        all_awards = np.zeros(board.shape)
        idx = np.random.choice(np.count_nonzero(board), 5)
        all_awards[tuple(map(lambda x: x[idx], np.where(board)))] = 1
        board[2][2] = 1  # locate human player
        board[7][7] = 1  # locate computer player

        return {
            'Board': board,
            'Human trace': human_trace,
            'Computer trace': computer_trace,
            'Human awards': human_awards,
            'Computer awards': computer_awards,
            'All awards': all_awards,
        }

    def get_random_valid_action(self, who):
        random_action = self.action_space.sample()
        while not self.valid_action(random_action, who):
            random_action = self.action_space.sample()
        return random_action

    def valid_action(self, action, who):
        if who == 'human':
            pos = np.where(self.dict_state['Human trace'] == 1)
        else:
            pos = np.where(self.dict_state['Computer trace'] == 1)
        next_pos = self.new_pos(pos, action)
        if self.dict_state['Board'][next_pos] == 0:
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
            assert True, "action: " + str(action) + " is not vallid at agent pos: " + str(
                current_pos)

    def move(self, action, agent):
        # assume action is valid
        if agent == 'human':
            current_pos = np.where(self.dict_state['Human trace'] == 1)
            self.dict_state['Human trace'] = self.dict_state['Human trace'] * 0.9
            next_pos = self.new_pos(current_pos, action)
            self.dict_state['Human trace'][next_pos] = 1
        elif agent == 'computer':
            current_pos = np.where(self.dict_state['Computer trace'] == 1)
            self.dict_state['Computer trace'] = self.dict_state['Computer trace'] * 0.9
            next_pos = self.new_pos(current_pos, action)
            self.dict_state['Computer trace'][next_pos] = 1
        else:
            assert True, "agent not define:" + str(agent)

    def predict_action(self, img):
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.human_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        action = np.argmax(score)
        #         print(
        #             "This image most likely belongs to {} with a {:.2f} percent confidence."
        #             .format(action, 100 * np.max(score))
        #         )
        return action

    def convertToImage(self, state):
        r = state['Human awards'] / 2 + state['Human trace']
        g = state['Board'] / 3 + state['All awards']
        b = state['Computer awards'] / 2 + state['Computer trace']
        rgb = np.dstack((r, g, b))
        return (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))  # NormalizeImage



# test env

# from stable_baselines.common.env_checker import check_env

# env = PackmanEnv()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)
