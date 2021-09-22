# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1
# https://pylessons.com/CartPole-DDQN/

import os
import random
import gym
import pylab
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop


def OurModel(input_shape, action_space):
    X_input = Input(shape=input_shape)
    X = X_input
    X = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer='he_uniform')(X)
    X = MaxPool2D()(X)
    X = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', kernel_initializer='he_uniform')(X)
    X = MaxPool2D()(X)
    X = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(X)
    X = MaxPool2D()(X)
    X = Flatten()(X)
    X = Dense(512, activation='relu', kernel_initializer='he_uniform')(X)
    X = Dense(256, activation='relu', kernel_initializer='he_uniform')(X)
    X = Dense(64, activation='relu', kernel_initializer='he_uniform')(X)
    # Output Layer with # of actions: 5 nodes (left, right, up, down, stay)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X)
    model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.999    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # defining model parameters
        self.ddqn = True
        self.Soft_Update = False

        self.TAU = 0.1 # target network soft update hyperparameter

        self.Save_Path = 'weights'
        self.scores, self.episodes, self.average = [], [], []
        
        if self.ddqn:
            print("----------Double DQN--------")
            self.Model_name = os.path.join(self.Save_Path,"ddqn_agent.h5")
        else:
            print("-------------DQN------------")
            self.Model_name = os.path.join(self.Save_Path,"dqn_agent.h5")
        
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
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def updateEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores) / len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episode', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = '_soft'
        try:
            pylab.savefig("data/images/"+dqn+self.env_name+softupdate+".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = 0
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                ep_rewards += reward
                if done:
                    # every step update target model
                    self.update_target_model()
                    # every episode, plot the result
                    average = self.PlotModel(ep_rewards, e)
                    print("episode: {}/{}, steps: {}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, ep_rewards, self.epsilon, average))
                    # decay epsilon
                    self.updateEpsilon()
                    
                self.replay()
        print("Saving trained model as ddqn_agent.h5")
        self.save("weights/ddqn_agent.h5")

    def test(self):
        self.load("ddqn_agent.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.expand_dims(next_state, axis=0)
                i += 1
                ep_rewards += reward
                if done:
                    print("episode: {}/{}, steps: {}, score: {}".format(e, self.EPISODES, i, ep_rewards))
                    break

if __name__ == "__main__":
    # env_name = 'gym_packman:Packman-v0'
    env_name = 'Breakout-v0'
    agent = DQNAgent(env_name)
    agent.run()
    agent.test()