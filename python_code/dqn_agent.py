import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')

from collections import deque
import numpy as np
import random
import gym
import pylab

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class Agent():
    def __init__(self, state_size, action_size, env):
        self.weight_backup      = "data/weights/dqn_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=20000)
        self.learning_rate      = 0.0001
        self.gamma              = 0.999
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.999
        self.brain              = self._build_model()
        self.env = env

        self.scores, self.episodes, self.average = [], [], []

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='elu'))
        model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='elu'))
        model.add(MaxPool2D())
        model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='elu'))
        model.add(Flatten())
        model.add(Dense(256, activation='elu'))
        model.add(Dense(64, activation='elu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        if os.path.isfile(self.weight_backup):
                    model.load_weights(self.weight_backup)
                    self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)
    
    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return self.env.get_random_valid_action('computer')
        
        act_values = self.brain.predict(state)
        act_values = np.squeeze(act_values, axis=0)
        action = np.argmax(act_values)
        while(not self.env.valid_action(action, 'computer')):
            act_values = np.delete(act_values, action)
            action = np.argmax(act_values)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=10, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores) / len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episode', fontsize=18)
        try:
            pylab.savefig("data/images/dqn_performance.png")
        except OSError:
            pass

        return str(self.average[-1])[:5]


sample_batch_size = 64
episodes = 1000
env_name = 'gym_packman:Packman-v0'
env = gym.make(env_name)
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = Agent(state_size, action_size, env)

ep_reward_list = []

for e in range(episodes):
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    ep_reward = 0
    i = 0
    while not done:
        # env.render()
        
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        i += 1
        # print(info)

    average = agent.PlotModel(ep_reward, e)
    agent.replay(sample_batch_size)
    print("episode: {}/{}, steps: {}, score: {}, e: {:.2}, average: {}".format(e, episodes, i, ep_reward, agent.exploration_rate, average))
    ep_reward_list.append(ep_reward)

agent.save_model()


# import numpy as np
# import gym

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf

# # ENV_NAME = 'CartPole-v0'
# ENV_NAME = 'gym_packman:Packman-v0'

# # Get the environment and extract the number of actions available in the Cartpole problem
# env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
# num_states = env.observation_space.shape
# num_actions = env.action_space.n

# def build_model(num_states, num_actions):
#     model = Sequential()
#     # Initialize weights between -3e-3 and 3-e3
#     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
#     model.add(Input(shape=(1,) + num_states))
                
#     model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding="same"))                
#     model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))
#     model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding="same"))
#     # model.add(MaxPool2D())

#     model.add(Flatten())
#     model.add(Dense(516, activation="elu", kernel_initializer=last_init))
#     model.add(Dense(64, activation="elu", kernel_initializer=last_init))
#     model.add(Dense(num_actions, activation="softmax", kernel_initializer=last_init))
#     return model

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory

# policy = EpsGreedyQPolicy()
# memory = SequentialMemory(limit=50000, window_length=1)

# model = build_model(num_states, num_actions)
# print(model.summary())

# dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
# target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
# dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# dqn.test(env, nb_episodes=5, visualize=True)