# import numpy as np
# import random

# import gym

# def simulate():
#     global epsilon, epsilon_decay
#     for episode in range(MAX_EPISODES):

#         # Init environment
#         state = env.reset()
#         total_reward = 0

#         # AI tries up to MAX_TRY times
#         for t in range(MAX_TRY):

#             # In the beginning, do random action to learn
#             if random.uniform(0, 1) < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = np.argmax(q_table[state])

#             # Do action and get result
#             next_state, reward, done, _ = env.step(action)
#             total_reward += reward

#             # Get correspond q value from state, action pair
#             q_value = q_table[state][action]
#             best_q = np.max(q_table[next_state])

#             # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
#             q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

#             # Set up for the next iteration
#             state = next_state

#             # Draw games
#             env.render()

#             # When episode is done, print reward
#             if done or t >= MAX_TRY - 1:
#                 print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
#                 break

#         # exploring rate decay
#         if epsilon >= 0.005:
#             epsilon *= epsilon_decay


# if __name__ == "__main__":
#     env = gym.make("gym_packman:Packman-v0")
#     MAX_EPISODES = 9999
#     MAX_TRY = 1000
#     epsilon = 1
#     epsilon_decay = 0.999
#     learning_rate = 0.1
#     gamma = 0.6
#     num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
#     q_table = np.zeros(num_box + (env.action_space.n,))
#     simulate()


# import base64
# import imageio
# import IPython
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import PIL.Image
# import pyvirtualdisplay

# import tensorflow as tf

# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import tf_metrics
# from tf_agents.networks import sequential
# from tf_agents.policies import random_tf_policy
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.trajectories import trajectory
# from tf_agents.specs import tensor_spec
# from tf_agents.utils import common

# num_iterations = 20000 # @param {type:"integer"}

# initial_collect_steps = 100  # @param {type:"integer"} 
# collect_steps_per_iteration = 1  # @param {type:"integer"}
# replay_buffer_max_length = 100000  # @param {type:"integer"}

# batch_size = 64  # @param {type:"integer"}
# learning_rate = 1e-3  # @param {type:"number"}
# log_interval = 200  # @param {type:"integer"}

# num_eval_episodes = 10  # @param {type:"integer"}
# eval_interval = 1000  # @param {type:"integer"}

# env_name = 'gym_packman:Packman-v0'
# env = suite_gym.load(env_name)

# train_py_env = suite_gym.load(env_name)
# eval_py_env = suite_gym.load(env_name)

# train_env = tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# fc_layer_params = (100, 50)
# action_tensor_spec = tensor_spec.from_spec(env.action_spec())
# num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# # Define a helper function to create Dense layers configured with the right
# # activation and kernel initializer.
# def dense_layer(num_units):
#   return tf.keras.layers.Dense(
#       num_units,
#       activation=tf.keras.activations.relu,
#       kernel_initializer=tf.keras.initializers.VarianceScaling(
#           scale=2.0, mode='fan_in', distribution='truncated_normal'))

# # QNetwork consists of a sequence of Dense layers followed by a dense layer
# # with `num_actions` units to generate one q_value per available action as
# # it's output.
# dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
# q_values_layer = tf.keras.layers.Dense(
#     num_actions,
#     activation=None,
#     kernel_initializer=tf.keras.initializers.RandomUniform(
#         minval=-0.03, maxval=0.03),
#     bias_initializer=tf.keras.initializers.Constant(-0.2))
# q_net = sequential.Sequential(dense_layers + [q_values_layer])

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# train_step_counter = tf.Variable(0)

# agent = dqn_agent.DqnAgent(
#     train_env.time_step_spec(),
#     train_env.action_spec(),
#     q_network=q_net,
#     optimizer=optimizer,
#     td_errors_loss_fn=common.element_wise_squared_loss,
#     train_step_counter=train_step_counter)

# agent.initialize()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')

# from collections import deque
# import numpy as np
# import random
# import gym

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
# from tensorflow.keras.optimizers import Adam

# class Agent():
#     def __init__(self, state_size, action_size, env):
#         self.weight_backup      = "cartpole_weight.h5"
#         self.state_size         = state_size
#         self.action_size        = action_size
#         self.memory             = deque(maxlen=2000)
#         self.learning_rate      = 0.001
#         self.gamma              = 0.95
#         self.exploration_rate   = 1.0
#         self.exploration_min    = 0.01
#         self.exploration_decay  = 0.995
#         self.brain              = self._build_model()
#         self.env = env

#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Conv2D(filters=6, kernel_size=(3,3),input_shape=self.state_size, activation='relu'))
#         model.add(Flatten())
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
#         # if os.path.isfile(self.weight_backup):
#         #             model.load_weights(self.weight_backup)
#         #             self.exploration_rate = self.exploration_min
#         return model

#     def save_model(self):
#             self.brain.save(self.weight_backup)
    
#     def act(self, state):
#         if np.random.rand() <= self.exploration_rate:
#             return self.env.get_random_valid_action('computer')
#         act_values = self.brain.predict(state)
#         act_values = np.squeeze(act_values, axis=0)
#         action = np.argmax(act_values)
#         while(not self.env.valid_action(action, 'computer')):
#             act_values = np.delete(act_values, action)
#             action = np.argmax(act_values)
#         return action

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
    
#     def replay(self, sample_batch_size):
#         if len(self.memory) < sample_batch_size:
#             return
#         sample_batch = random.sample(self.memory, sample_batch_size)
#         for state, action, reward, next_state, done in sample_batch:
#             target = reward
#             if not done:
#               target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
#             target_f = self.brain.predict(state)
#             target_f[0][action] = target
#             self.brain.fit(state, target_f, epochs=1, verbose=0)
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay


# class CartPole:
#     def __init__(self):
#         self.sample_batch_size = 32
#         self.episodes          = 10000
#         self.env_name = 'gym_packman:Packman-v0'
#         # self.env_name = 'CartPole-v1'
#         self.env               = gym.make(self.env_name)
#         self.state_size        = self.env.observation_space.shape
#         self.action_size       = self.env.action_space.n
#         self.agent             = Agent(self.state_size, self.action_size, self.env)
    
#     def run(self):
#         try:
#             for index_episode in range(self.episodes):
#                 state = self.env.reset()
#                 state = np.expand_dims(state, axis=0)
#                 done = False
#                 index = 0
#                 while not done:
#                     self.env.render()
#                     action = self.agent.act(state)
#                     next_state, reward, done, _ = self.env.step(action)
#                     next_state = np.expand_dims(next_state, axis=0)
#                     self.agent.remember(state, action, reward, next_state, done)
#                     state = next_state
#                     index += 1
#                 print("Episode {}# Score: {}".format(index_episode, index + 1))
#                 self.agent.replay(self.sample_batch_size)
#         finally:
#             self.agent.save_model()

# if __name__ == "__main__":
#     cartpole = CartPole()
#     cartpole.run()

import numpy as np
import gym

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'gym_packman:Packman-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
num_states = env.observation_space.shape
# num_states = (1,1,10,10,3)
num_actions = env.action_space.n

print(env.reset().shape)
print(env.step(0)[0].shape)

def build_model(num_states, num_actions):
    model = Sequential()
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    model.add(Input(shape=(1,) + num_states))
                
    model.add(Conv2D(filters=7, kernel_size=(3,3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=2, strides=1))
                
    model.add(Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=2, strides=1))

    model.add(Flatten())
    model.add(Dense(num_actions, activation="softmax", kernel_initializer=last_init))
    return model

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

model = build_model(num_states, num_actions)
print(model.summary())

dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
dqn.fit(env, nb_steps=5000, visualize=True, verbose=2)

dqn.test(env, nb_episodes=5, visualize=True)