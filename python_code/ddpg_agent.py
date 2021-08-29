import gym
import numpy as np

# randomize numbers:
import random

# ignore warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

# ## added this lines to solve GPU errors!
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

sim = 'gym_packman:Packman-v0'
env = gym.make(sim)

num_states = env.observation_space.shape
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.n
print("Size of Action Space ->  {}".format(num_actions))

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element  
        self.state_buffer = np.zeros((self.buffer_capacity,) + num_states)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity,) + num_states)


    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        # self.state_buffer[index] = tf.expand_dims(obs_tuple[0],-1)
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        # self.next_state_buffer[index] = tf.expand_dims(obs_tuple[3],-1)
        self.next_state_buffer[index] = obs_tuple[3]
        

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )
        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # print("state_batch: ", state_batch.shape)
        # print("action_batch: ", action_batch.shape)
        # print("reward_batch: ", reward_batch.shape)
        # print("next_state_batch: ", next_state_batch.shape)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.
Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)

    inputs = layers.Input(shape=num_states)

    out = layers.Conv2D(filters=16, kernel_size=(3,3),input_shape=num_states, activation='relu',kernel_initializer=last_init)(inputs)
    out = layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=num_states, activation='relu',kernel_initializer=last_init)(out)
    out = layers.MaxPool2D()(out)
    out = layers.Dropout(rate=0.5)(out)

    out = tf.keras.layers.Flatten()(out)

    outputs = layers.Dense(32, activation="relu",kernel_initializer=last_init)(out)
    outputs = layers.Dense(10, activation="relu",kernel_initializer=last_init)(outputs)

    outputs = layers.Dense(num_actions, activation="relu")(outputs)
    model = tf.keras.Model(inputs, outputs)

    # model = Sequential([
    #     layers.experimental.preprocessing.Rescaling(1./255, input_shape=num_states),
    #     layers.Conv2D(16, 3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.Conv2D(32, 3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    #     # layers.MaxPooling2D(),
    #     layers.Dropout(0.5),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
    #     layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
    #     layers.Dense(num_actions)
    #     ])
    
    return model

def get_critic():
    
    # State as input
    state_input = layers.Input(shape=num_states)

    state_out = layers.Conv2D(filters=16, kernel_size=(3,3),input_shape=num_states, activation='relu')(state_input)    
    state_out = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(state_out)
    state_out = layers.MaxPool2D()(state_out)
    state_out = layers.Dropout(rate=0.5)(state_out)
    
    state_out = tf.keras.layers.Flatten()(state_out)

    state_out = layers.Dense(32, activation="relu")(state_out)
    state_out = layers.Dense(10, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=num_actions)
    action_out = layers.Dense(10, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(10, activation="relu")(concat)
    outputs = layers.Dense(1)(out)
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""

def policy(state, epsilon, episode):
    
    # exploration using epsilon greedy which decay over time:
    t = np.max([episode ,env.step_num])
    t = 1000000
    if epsilon/(t**0.5) >= random.uniform(0, 1):
            print(f"{episode}) Exploration!")
            action = env.get_random_valid_action('computer')
    else:
        sampled_actions = tf.squeeze(actor_model(state)).numpy()
        print(f'{episode}) sampled_actions: {sampled_actions}')

        action = np.argmax(sampled_actions)
        # We make sure action is within bounds
        while(not env.valid_action(action, 'computer')):
            print(f'{episode}) best action not valid ====> {action}')
            sampled_actions = np.delete(sampled_actions, action)
            action = np.argmax(sampled_actions)
    
    return action

"""
## Training hyperparameters
"""
# exploration:
epsilon = 3

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# try to load weights if weights file exist (for continous training):
try:
    actor_model.load_weights('./weights/packman_actor.h5')
    critic_model.load_weights('./weights/packman_critic.h5')
except:
    print("-------------- no weights loaded! -------------------")

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 4
# Discount factor for future rewards
gamma = 0.995
# Used to update target networks
tau = 0.005

buffer = Buffer(10000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []

# try to load rewards history if rewards file exist (for continous training):
try:
    ep_reward_list = np.load('./weights/rewards.npy')
    ep_reward_list = ep_reward_list.tolist()
except:
    print("-------------- no rewards history loaded! -------------------")

for ep in range(1,total_episodes+1):

    prev_state = env.reset() 
    episodic_reward = 0
    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, epsilon, ep)
        print(f"{ep}) action: {action}")

        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)
        print(f"{ep}) reward: {reward}")
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state
        # raise ValueError()

    ep_reward_list.append(episodic_reward)

    # Reward of last episode
    print("Episode * {} * Reward is ==> {}".format(ep, episodic_reward))

    # # Save the weights after 10 steps:
    # if ep % 10 == 0:
    #     print("Weights saved")
    #     actor_model.save_weights("./weights/lgsvl_actor.h5")
    #     critic_model.save_weights("./weights/lgsvl_critic.h5")

    #     target_actor.save_weights("./weights/lgsvl_target_actor.h5")
    #     target_critic.save_weights("./weights/lgsvl_target_critic.h5")

    #     # Ido: Heavy operation, need to think something else
    #     # Save all rewards for next run
    #     np.save('./weights/rewards.npy', ep_reward_list)



# Plotting graph
# Episodes versus Avg. Rewards
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # plt.savefig('./images/performance.png')
plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.
Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.
"""