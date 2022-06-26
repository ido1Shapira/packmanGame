# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1
# https://pylessons.com/CartPole-DDQN/

random_seed = 0
from audioop import avg
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
# import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
import time

# if setting seed the result is always the same
# np.random.seed(random_seed)
# random.seed(random_seed)
# tf.random.set_seed(random_seed)


def OurModel(input_shape, action_space):
    X_input = Input(shape=input_shape)
    X = X_input
    X = Conv2D(filters=4, kernel_size=(4,4), padding='same', activation='relu')(X)
    X = Conv2D(filters=8, kernel_size=(4,4), padding='same', activation='relu')(X)
    X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
    X = MaxPool2D()(X)
    X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
    X = Flatten()(X)
    # X = Dense(256, activation='relu')(X)
    X = Dense(32, activation='relu')(X)
    # Output Layer with # of actions: 5 nodes (left, right, up, down, stay)
    X = Dense(action_space, activation="linear")(X)

    model = Model(inputs = X_input, outputs = X)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.0002), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self, env_name, map, beta):
        self.map_dir = map

        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(random_seed)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        # defining SARL parameters
        self.beta = beta
        
        # create main model
        self.model = OurModel(input_shape=self.state_size, action_space = self.action_size)

    def load(self, name):
        self.model = load_model(name)

    def test(self, test_episodes, path):
        score_results = []
        step_results = []
        sarl_score_results = []

        self.load(path)

        for e in range(test_episodes):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = self.env.rewards['Start']
            ep_SARL_rewards = self.env.rewards['Start']
            while not done:
                # self.env.render(mode='human')
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                SARL_reward = self.beta * reward + (1 - self.beta) * info['human_reward']
                state = np.expand_dims(next_state, axis=0)
                i += 1
                ep_rewards += reward
                ep_SARL_rewards += SARL_reward
                
                # print(info)  
                # time.sleep(0.5)
                
                if done:
                    print("episode: {}/{}, steps: {}, score: {}, SARL score: {}".format(e, test_episodes, i, ep_rewards, ep_SARL_rewards))
                    score_results.append(ep_rewards)
                    sarl_score_results.append(ep_SARL_rewards)
                    step_results.append(i)
                    break
        
        return np.array(step_results), np.array(score_results), np.array(sarl_score_results)

if __name__ == "__main__":
    env_name = 'gym_packman:Packman-v0'
    dir_map = 'map 5'
    beta = 0.615

    num = 1000
    avg_list = []
    for i in range(6):
        humanModel_version = '_v' + str(i)
        model_path = "data/"+dir_map+"/weights/ddqn_agent_distribution"+humanModel_version+".h5"

        agent = DQNAgent(env_name, dir_map, beta)
        step_results, score_results, sarl_score_results = agent.test(num, model_path)
        avg_list.append({humanModel_version: (step_results.mean(), score_results.mean(), sarl_score_results.mean())})

    model_path = "data/"+dir_map+"/weights/SARL_ddqn_agent_"+str(beta)+"_distribution.h5"
    sarl = DQNAgent(env_name, dir_map, beta)
    step_results, score_results, sarl_score_results = sarl.test(num, model_path)
    avg_list.append({'sarl': (step_results.mean(), score_results.mean(), sarl_score_results.mean())})
    
    print(avg_list)