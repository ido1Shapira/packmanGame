# # https://github.com/keras-team/keras-io/blob/master/examples/rl/ppo_cartpole.py
# """
# Title: Proximal Policy Optimization
# """

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
# import gym
# import scipy.signal
# # import matplotlib.pyplot as plt
# import pylab as plt

# """
# ## Functions and class
# """

# def discounted_cumulative_sums(x, discount):
#     # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# class Buffer:
#     # Buffer for storing trajectories
#     def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
#         # Buffer initialization
#         self.observation_buffer = np.zeros(
#             (size,) + observation_dimensions, dtype=np.float32
#         )
#         self.action_buffer = np.zeros(size, dtype=np.int32)
#         self.advantage_buffer = np.zeros(size, dtype=np.float32)
#         self.reward_buffer = np.zeros(size, dtype=np.float32)
#         self.return_buffer = np.zeros(size, dtype=np.float32)
#         self.value_buffer = np.zeros(size, dtype=np.float32)
#         self.logprobability_buffer = np.zeros(size, dtype=np.float32)
#         self.gamma, self.lam = gamma, lam
#         self.pointer, self.trajectory_start_index = 0, 0

#     def store(self, observation, action, reward, value, logprobability):
#         # Append one step of agent-environment interaction
#         self.observation_buffer[self.pointer] = observation
#         self.action_buffer[self.pointer] = action
#         self.reward_buffer[self.pointer] = reward
#         self.value_buffer[self.pointer] = value
#         self.logprobability_buffer[self.pointer] = logprobability
#         self.pointer += 1

#     def finish_trajectory(self, last_value=0):
#         # Finish the trajectory by computing advantage estimates and rewards-to-go
#         path_slice = slice(self.trajectory_start_index, self.pointer)
#         rewards = np.append(self.reward_buffer[path_slice], last_value)
#         values = np.append(self.value_buffer[path_slice], last_value)

#         deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

#         self.advantage_buffer[path_slice] = discounted_cumulative_sums(
#             deltas, self.gamma * self.lam
#         )
#         self.return_buffer[path_slice] = discounted_cumulative_sums(
#             rewards, self.gamma
#         )[:-1]

#         self.trajectory_start_index = self.pointer

#     def get(self):
#         # Get all data of the buffer and normalize the advantages
#         self.pointer, self.trajectory_start_index = 0, 0
#         advantage_mean, advantage_std = (
#             np.mean(self.advantage_buffer),
#             np.std(self.advantage_buffer),
#         )
#         self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
#         return (
#             self.observation_buffer,
#             self.action_buffer,
#             self.advantage_buffer,
#             self.return_buffer,
#             self.logprobability_buffer,
#         )


# def logprobabilities(logits, a):
#     # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
#     logprobabilities_all = tf.nn.log_softmax(logits)
#     logprobability = tf.reduce_sum(
#         tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
#     )
#     return logprobability


# # Sample action from actor
# @tf.function
# def sample_action(observation):
#     logits = actor(observation)
#     action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
#     return logits, action


# # Train the policy by maxizing the PPO-Clip objective
# @tf.function
# def train_policy(
#     observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
# ):

#     with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
#         ratio = tf.exp(
#             logprobabilities(actor(observation_buffer), action_buffer)
#             - logprobability_buffer
#         )
#         min_advantage = tf.where(
#             advantage_buffer > 0,
#             (1 + clip_ratio) * advantage_buffer,
#             (1 - clip_ratio) * advantage_buffer,
#         )

#         policy_loss = -tf.reduce_mean(
#             tf.minimum(ratio * advantage_buffer, min_advantage)
#         )
#     policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
#     policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

#     kl = tf.reduce_mean(
#         logprobability_buffer
#         - logprobabilities(actor(observation_buffer), action_buffer)
#     )
#     kl = tf.reduce_sum(kl)
#     return kl


# # Train the value function by regression on mean-squared error
# @tf.function
# def train_value_function(observation_buffer, return_buffer):
#     with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
#         value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
#     value_grads = tape.gradient(value_loss, critic.trainable_variables)
#     value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

# def PlotModel(score, step, episode):
#     window_size = 50
#     scores.append(score)
#     episodes.append(episode)        
#     steps.append(step)
#     if len(steps) > window_size:
#         # moving avrage:
#         averages.append(sum(scores[-1 * window_size: ]) / window_size)
#         averages_steps.append(sum(steps[-1 * window_size: ]) / window_size)
#     else:
#         averages.append(sum(scores) / len(scores))
#         averages_steps.append(sum(steps) / len(steps))

#     ax1.plot(scores, 'b')
#     ax1.plot(averages, 'r')
#     ax2.plot(steps, 'b')
#     ax2.plot(averages_steps, 'r')
#     try:
#         plt.savefig("data/images/"+"ppo_"+'gym_packman:Packman-v0'+".png")
#         # plt.show()
#     except OSError:
#         print('im here')
#         pass

#     return str(averages[-1])[:5]

# """
# ## Hyperparameters
# """

# # Hyperparameters of the PPO algorithm
# steps_per_epoch = 300
# epochs = 1000
# gamma = 0.99
# clip_ratio = 0.1
# policy_learning_rate = 3e-5
# value_function_learning_rate = 1e-4
# train_policy_iterations = 128
# train_value_iterations = 128
# lam = 0.97
# target_kl = 0.01

# # True if you want to train the agent
# train = True

# """
# ## Initializations
# """

# # Initialize the environment and get the dimensionality of the
# # observation space and the number of possible actions
# env = gym.make('gym_packman:Packman-v0')
# observation_dimensions = env.observation_space.shape
# num_actions = env.action_space.n

# # Initialize the buffer
# buffer = Buffer(observation_dimensions, steps_per_epoch, gamma, lam)

# # Initialize the actor and the critic as keras models
# def get_actor(state_size, action_size):
#     X_input = Input(shape=state_size)
#     X = X_input
#     X = Conv2D(filters=4, kernel_size=(4,4), padding='same', activation='relu')(X)
#     X = Conv2D(filters=8, kernel_size=(4,4), padding='same', activation='relu')(X)
#     X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
#     X = MaxPool2D()(X)
#     X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
#     X = Flatten()(X)
#     # X = Dense(256, activation='relu')(X)
#     X = Dense(32, activation='relu')(X)
#     # Output Layer with # of actions: 5 nodes (left, right, up, down, stay)
#     X = Dense(action_size, activation="linear")(X)
#     model = Model(inputs = X_input, outputs = X)
#     # model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.policy_learning_rate), metrics=["accuracy"])
#     model.summary()
#     return model

# def get_critic(state_size):
#     X_input = Input(shape=state_size)
#     X = X_input
#     X = Conv2D(filters=4, kernel_size=(4,4), padding='same', activation='relu')(X)
#     X = Conv2D(filters=8, kernel_size=(4,4), padding='same', activation='relu')(X)
#     X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
#     X = MaxPool2D()(X)
#     X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
#     X = Flatten()(X)
#     # X = Dense(256, activation='relu')(X)
#     X = Dense(32, activation='relu')(X)
#     # Output Layer with # of actions: 5 nodes (left, right, up, down, stay)
#     X = Dense(1, activation="linear", kernel_initializer='he_uniform')(X)
#     model = Model(inputs = X_input, outputs = X)
#     # model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.value_function_learning_rate), metrics=["accuracy"])
#     model.summary()
#     return model

# actor = get_actor(observation_dimensions, num_actions)
# critic = get_critic(observation_dimensions)

# if train:
#     # Initialize the policy and the value function optimizers
#     policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
#     value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

#     # Initialize the observation, episode return and episode length
#     observation, episode_return, episode_length = env.reset(), 0, 0

#     scores, steps, episodes, averages, averages_steps = [], [], [], [], []
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9))
#     ax1.set_ylabel('Score', fontsize=15)
#     ax2.set_ylabel('Step', fontsize=15)
#     ax2.set_xlabel('Episode', fontsize=15)

#     """
#     ## Train
#     """
#     # Iterate over the number of epochs
#     for epoch in range(epochs):
#         # Initialize the sum of the returns, lengths and number of episodes for each epoch
#         sum_return = 0
#         sum_length = 0
#         num_episodes = 0

#         # Iterate over the steps of each epoch
#         for t in range(steps_per_epoch):
#             # Get the logits, action, and take one step in the environment
#             observation = tf.expand_dims(observation, axis=0)
#             logits, action = sample_action(observation)
#             observation_new, reward, done, _ = env.step(action[0].numpy())
#             episode_return += reward
#             episode_length += 1

#             # Get the value and log-probability of the action
#             value_t = critic(observation)
#             logprobability_t = logprobabilities(logits, action)
#             # Store obs, act, rew, v_t, logp_pi_t
#             buffer.store(observation, action, reward, value_t, logprobability_t)

#             # Update the observation
#             observation = observation_new

#             # Finish trajectory if reached to a terminal state
#             terminal = done
#             if terminal or (t == steps_per_epoch - 1):
#                 last_value = 0 if done else critic(tf.expand_dims(observation, axis=0))
#                 buffer.finish_trajectory(last_value)
#                 sum_return += episode_return
#                 sum_length += episode_length
#                 num_episodes += 1
#                 average = PlotModel(episode_return, t + 1, epoch)
#                 # Print mean return and length for each epoch
#                 print("episode: {}/{}, steps: {}, score: {}, average: {}".format(epoch, epochs, t + 1, episode_return, average))
#                 observation, episode_return, episode_length = env.reset(), 0, 0

#         # Get values from the buffer
#         (
#             observation_buffer,
#             action_buffer,
#             advantage_buffer,
#             return_buffer,
#             logprobability_buffer,
#         ) = buffer.get()

#         # Update the policy and implement early stopping using KL divergence
#         for _ in range(train_policy_iterations):
#             kl = train_policy(
#                 observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
#             )
#             if kl > 1.5 * target_kl:
#                 # Early Stopping
#                 break

#         # Update the value function
#         for _ in range(train_value_iterations):
#             train_value_function(observation_buffer, return_buffer)


#     print("\nFinish training, here some statistics:\n")
#     print("Mean average score: ", np.mean(averages))
#     print("Mean average steps: ", np.mean(averages_steps))

#     print("\nMin score: ", np.min(scores))
#     print("Min steps: ", np.min(steps))
#     print("\nMax score: ", np.max(scores))
#     print("Max steps: ", np.max(steps))

#     print("Saving trained model as ppo_agent.h5")
#     actor.save("weights/ppo_actor_agent.h5")
#     critic.save("weights/ppo_critic_agent.h5")

# else: # test the agent
#     actor = load_model("weights/ppo_actor_agent.h5")
#     critic = load_model("weights/ppo_critic_agent.h5")
#     test_episodes = 5
#     for e in range(test_episodes):
#         state = env.reset()
#         state = np.expand_dims(state, axis=0)
#         done = False
#         i = 0
#         ep_rewards = 0
#         while not done:
#             env.render()
#             _ , action = sample_action(state)
#             next_state, reward, done, info = env.step(action[0].numpy())
#             state = np.expand_dims(next_state, axis=0)
#             i += 1
#             ep_rewards += reward
#             print(info)
#             if done:
#                 print("episode: {}/{}, steps: {}, score: {}".format(e, test_episodes, i, ep_rewards))
#                 break

# https://github.com/keras-team/keras-io/blob/master/examples/rl/ppo_cartpole.py
"""
Title: Proximal Policy Optimization
"""
import numpy as np
from numpy.lib.function_base import average
import tensorflow as tf
import gym
import scipy.signal
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
"""
## Functions and class
"""
def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size,) + observation_dimensions, dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )

class PPOAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        # Hyperparameters of the PPO algorithm
        self.steps_per_epoch = 300
        self.epochs = 1000
        self.gamma = 0.99
        self.clip_ratio = 0.1
        self.policy_learning_rate = 3e-5
        self.value_function_learning_rate = 1e-4
        self.train_policy_iterations = 128
        self.train_value_iterations = 128
        self.lam = 0.97
        self.target_kl = 0.01


        # policy_decayed_lr = ExponentialDecay(self.policy_learning_rate, decay_steps=self.epochs, decay_rate=0.99, staircase=True)
        # value_decayed_lr = ExponentialDecay(self.policy_learning_rate, decay_steps=self.epochs, decay_rate=0.99, staircase=True)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = Adam(learning_rate=self.value_function_learning_rate)

        # Initialize the buffer
        self.buffer = Buffer(self.state_size, self.steps_per_epoch)
        
        self.scores, self.steps, self.episodes, self.averages, self.averages_steps = [], [], [], [], []
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(18, 9))
        self.ax1.set_ylabel('Score', fontsize=15)
        self.ax2.set_ylabel('Step', fontsize=15)
        self.ax2.set_xlabel('Episode', fontsize=15)

        self.actor = self.get_actor()
        self.critic = self.get_critic()

    def get_actor(self):
        X_input = Input(shape=self.state_size)
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
        X = Dense(self.action_size, activation="linear")(X)
        model = Model(inputs = X_input, outputs = X)
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.policy_learning_rate), metrics=["accuracy"])
        return model

    def get_critic(self):
        X_input = Input(shape=self.state_size)
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
        X = Dense(1, activation="linear", kernel_initializer='he_uniform')(X)
        model = Model(inputs = X_input, outputs = X)
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.value_function_learning_rate), metrics=["accuracy"])
        return model

    def load(self, actor_name, critic_name):
        self.actor = load_model(actor_name)
        self.critic = load_model(critic_name)

    def save(self, actor_name, critic_name):
        print("\nFinish training, here some statistics:\n")
        print("Mean average score: ", np.mean(self.averages))
        print("Mean average steps: ", np.mean(self.averages_steps))
        
        print("\nMin score: ", np.min(self.scores))
        print("Min steps: ", np.min(self.steps))
        print("\nMax score: ", np.max(self.scores))
        print("Max steps: ", np.max(self.steps))

        print("Saving trained model as ppo_agent.h5")
        self.actor.save(actor_name)
        self.critic.save(critic_name)

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
        try:
            plt.savefig("data/images/"+"ppo_"+self.env_name+".png")
        except OSError:
            pass

        return str(self.averages[-1])[:5]

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.action_size) * logprobabilities_all, axis=1
        )
        return logprobability

    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self,
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


    def run(self):
        observation = self.env.reset()
        for epoch in range(self.epochs):
            episode_return = 0
            # Iterate over the steps of each epoch
            for t in range(self.steps_per_epoch):
                # Get the logits, action, and take one step in the environment
                observation = tf.expand_dims(observation, axis=0)
                logits, action = self.sample_action(observation)
                observation_new, reward, done, _ = self.env.step(action[0].numpy())
                episode_return += reward
                # Get the value and log-probability of the action
                value_t = self.critic(observation)
                logprobability_t = self.logprobabilities(logits, action)
                # Store obs, act, rew, v_t, logp_pi_t
                self.buffer.store(observation, action, reward, value_t, logprobability_t)
                # Update the observation
                observation = observation_new
                
                if done or (t == self.steps_per_epoch - 1):
                    last_value = 0 if done else self.critic(tf.expand_dims(observation, axis=0))
                    self.buffer.finish_trajectory(last_value)
                    # every episode, plot the result
                    average = self.PlotModel(episode_return, t + 1, epoch)
                    print("episode: {}/{}, steps: {}, score: {}, average: {}".format(epoch, self.epochs, t + 1, episode_return, average))
                    observation, episode_return = self.env.reset(), 0
                    break

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()
            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)
        
        self.save("weights/ppo_actor_agent.h5", "weights/ppo_critic_agent.h5")

    def test(self, test_episodes):
        self.load("weights/ppo_actor_agent.h5", "weights/ppo_critic_agent.h5")
        for e in range(test_episodes):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = 0
            while not done:
                self.env.render()
                _ , action = self.sample_action(state)
                next_state, reward, done, _ = self.env.step(action[0].numpy())
                state = np.expand_dims(next_state, axis=0)
                i += 1
                ep_rewards += reward
                if done:
                    print("episode: {}/{}, steps: {}, score: {}".format(e, test_episodes, i, ep_rewards))
                    break


if __name__ == "__main__":
    env_name = 'gym_packman:Packman-v0'
    agent = PPOAgent(env_name)
    agent.run()
    agent.test(5)