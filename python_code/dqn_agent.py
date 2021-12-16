# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1
# https://pylessons.com/CartPole-DDQN/

random_seed = 0

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import tensorflow as tf
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
    def __init__(self, env_name):
        self.map_dir = 'map 2'

        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(random_seed)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        self.EPISODES = 1010 #900
        self.memory = deque(maxlen=100000)
        
        self.gamma = 0.999 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.997
        self.batch_size = 128
        self.train_start = 2000 # memory_size

        # defining model parameters
        self.ddqn = True
        self.Soft_Update = True

        self.TAU = 0.1 # target network soft update hyperparameter

        # defining SARL parameters
        self.beta = 0.3

        self.scores, self.steps, self.episodes, self.averages, self.averages_steps = [], [], [], [], []
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(18, 9))
        self.ax1.set_ylabel('Score', fontsize=15)
        self.ax2.set_ylabel('Step', fontsize=15)
        self.ax2.set_xlabel('Episode', fontsize=15)
        self.ax1.set_ylim([-2, 1.5])
        self.ax2.set_ylim([0, 100])
        
        # create main model
        self.model = OurModel(input_shape=self.state_size, action_space = self.action_size)
        self.target_model = OurModel(input_shape=self.state_size, action_space = self.action_size)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            # return self.env.get_random_valid_action('computer')
        else:
            return np.argmax(self.model.predict(state))
            # act_values = self.model.predict(state)
            # act_values = np.squeeze(act_values, axis=0)
            # action = np.argmax(act_values)
            # while(not self.env.valid_action(action, 'computer')):
            #     act_values = np.delete(act_values, action)
            #     action = np.argmax(act_values)
            # return action

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))
        
        state = np.zeros((self.batch_size,) + self.state_size)
        next_state = np.zeros((self.batch_size,) + self.state_size)
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target,  epochs=2, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def updateEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        print("\nFinish training, here some statistics:\n")
        print("Mean score: ", np.mean(self.averages))
        print("Mean steps: ", np.mean(self.averages_steps))
        print("\nMin score: ", np.min(self.scores))
        print("Min steps: ", np.min(self.steps))
        print("\nMax score: ", np.max(self.scores))
        print("Max steps: ", np.max(self.steps))

        self.model.save(name)
    
    def PlotModel(self, score, step, episode):
        window_size = 50 #int(self.epochs / 100)
        self.scores.append(score)
        self.episodes.append(episode)        
        self.steps.append(step)
        if len(self.steps) > window_size:
            # moving avrage:
            self.averages.append(sum(self.scores[-1 * window_size: ]) / window_size)
            self.averages_steps.append(sum(self.steps[-1 * window_size: ]) / window_size)
        else:
            self.averages.append(sum(self.scores) / len(self.scores))
            self.averages_steps.append(sum(self.steps) / len(self.steps))

        self.ax1.plot(self.scores, 'b')
        self.ax1.plot(self.averages, 'r')
        self.ax2.plot(self.steps, 'b')
        self.ax2.plot(self.averages_steps, 'r')

        dqn = 'DQN_'
        softupdate = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = 'soft'
        try:
            plt.savefig("data/"+self.map_dir+"/images/"+dqn+softupdate+".png", dpi = 150)
        except OSError:
            pass

        return str(self.averages[-1])[:5]

    def run(self):
        n_dones = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = self.env.rewards['Start']
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                ep_rewards += reward
                if done:
                    if i < 300:
                        n_dones += 1 
                    # every step update target model
                    self.update_target_model()
                    # every episode, plot the result
                    average = self.PlotModel(ep_rewards, i, e)
                    print("episode: {}/{}, steps: {}, score: {}, e: {:.2}, average: {}, dones: {}".format(e, self.EPISODES, i, ep_rewards, self.epsilon, average, n_dones))
                    # decay epsilon
                    self.updateEpsilon()
                    
                self.replay()
        self.save("data/"+self.map_dir+"/weights/ddqn_agent.h5")

    def test(self, test_episodes):
        self.load("data/"+self.map_dir+"/weights/ddqn_agent.h5")
        for e in range(test_episodes):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = self.env.rewards['Start']
            ep_SARL_rewards = self.env.rewards['Start']
            while not done:
                self.env.render(mode='human')
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                SARL_reward = self.beta * reward + (1 - self.beta) * info['human_reward']
                state = np.expand_dims(next_state, axis=0)
                i += 1
                ep_rewards += reward
                ep_SARL_rewards += SARL_reward
                # print(info)
                
                time.sleep(0.5)
                
                if done:
                    print("episode: {}/{}, steps: {}, score: {}, SARL score: {}".format(e, test_episodes, i, ep_rewards, ep_SARL_rewards))
                    break

if __name__ == "__main__":
    env_name = 'gym_packman:Packman-v0'
    agent = DQNAgent(env_name)
    agent.run()
    agent.test(5)