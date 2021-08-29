import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')

from collections import deque
import numpy as np
import random
import gym

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

class Agent():
    def __init__(self, state_size, action_size, env):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()
        self.env = env

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3,3),input_shape=self.state_size, activation='relu'))
        model.add(Flatten())
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        # if os.path.isfile(self.weight_backup):
        #             model.load_weights(self.weight_backup)
        #             self.exploration_rate = self.exploration_min
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
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 10000
        self.env_name = 'gym_packman:Packman-v0'
        # self.env_name = 'CartPole-v1'
        self.env               = gym.make(self.env_name)
        self.state_size        = self.env.observation_space.shape
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size, self.env)
    
    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.expand_dims(state, axis=0)
                done = False
                index = 0
                while not done:
                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.expand_dims(next_state, axis=0)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()


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
# # num_states = (1,1,10,10,3)
# num_actions = env.action_space.n

# print(env.reset().shape)
# print(env.step(0)[0].shape)

# def build_model(num_states, num_actions):
#     model = Sequential()
#     # Initialize weights between -3e-3 and 3-e3
#     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
#     model.add(Input(shape=(1,) + num_states))
                
#     model.add(Conv2D(filters=6, kernel_size=(3,3), activation='relu', padding="same"))                
#     model.add(Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding="same"))
#     model.add(MaxPool2D(pool_size=2, strides=1))

#     model.add(Flatten())
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