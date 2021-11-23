import gym
from gym.spaces import Discrete, Box
from multiprocessing import Process, Pipe
import numpy as np
import pickle
import cloudpickle

class SubprocVecEnv():
    def __init__(self, env_fns):
        self.action_space = Discrete(5)

        # Define a 2-D observation space
        # self.observation_shape = (10, 10, 6)
        self.observation_shape = (10, 10, 3)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                     high=np.ones(self.observation_shape),
                                     dtype=np.float32)

        self.waiting = False
        self.closed = False
        self.no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(self.no_of_envs)])
        self.ps = []
		
        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target = worker, 
                args = (wrk, rem, CloudpickleWrapper(fn)))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()
        	
    def step_async(self, actions):
        if self.waiting:
            raise AlreadySteppingError
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
	
    def step_wait(self):
        if not self.waiting:
            raise NotSteppingError
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
	
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
	
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        return np.stack([remote.recv() for remote in self.remotes])
	
    def render(self):
        # render not working
        raise RuntimeError('render not working')
        for remote in self.remotes:
            remote.send(('render', None))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
    
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))

        elif cmd == 'render':
            remote.send(env.render())
        
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError(f'command: {cmd} not in list')

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)

class CloudpickleWrapper(object):
	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		self.x = pickle.loads(ob)
	
	def __call__(self):
		return self.x()

def make_mp_envs(env_id, num_env, seed, start_idx = 0):
	def make_env(rank):
		def fn():
			env = gym.make(env_id)
			env.seed(seed + rank)
			return env
		return fn
	return SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])

import random
import time

# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1
# https://pylessons.com/CartPole-DDQN/

import os
import random
import gym
# import pylab
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop


def OurModel(input_shape, action_space):
    X_input = Input(shape=input_shape)
    X = X_input
    X = Conv2D(filters=4, kernel_size=(4,4), padding='same', activation='relu')(X)
    X = Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu')(X)
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
    def __init__(self, env_name, num_envs=5):
        self.env_name = env_name       
        self.env = make_mp_envs(self.env_name, num_envs, 0)
        # self.env.seed(0)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n

        self.EPISODES = 1500 #1010
        self.memory = deque(maxlen=100000)
        
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998 #0.997
        self.batch_size = 128
        self.train_start = 2000 # memory_size

        # defining model parameters
        self.ddqn = True
        self.Soft_Update = True

        self.TAU = 0.1 # target network soft update hyperparameter

        self.Save_Path = 'weights'
        self.scores, self.steps, self.episodes, self.averages, self.averages_steps = [], [], [], [], []
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(18, 9))
        self.ax1.set_ylabel('Score', fontsize=15)
        self.ax2.set_ylabel('Step', fontsize=15)
        self.ax2.set_xlabel('Episode', fontsize=15)
        
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

        print("Saving trained model as ppo_agent.h5")
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
            softupdate = '_soft'
        try:
            plt.savefig("data/images/"+dqn+self.env_name+softupdate+".png")
        except OSError:
            pass

        return str(self.averages[-1])[:5]

    def run(self):
        n_dones = 0
        epoce_time = 0.001 # hours
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = 0
            t_end = time.time() + 60 * 60 * epoce_time
            while time.time() <= t_end:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                ep_rewards += reward
            
            # every step update target model
            self.update_target_model()
            # every episode, plot the result
            average = self.PlotModel(ep_rewards, i, e)
            print("episode: {}/{}, steps: {}, score: {}, e: {:.2}, average: {}, dones: {}".format(e, self.EPISODES, i, ep_rewards, self.epsilon, average, n_dones))
            # decay epsilon
            self.updateEpsilon()
                    
            self.replay()

        self.save("weights/ddqn_agent.h5")

    def test(self, test_episodes):
        self.load("weights/ddqn_agent.h5")
        for e in range(test_episodes):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            i = 0
            ep_rewards = 0
            while not done:
                self.env.render(mode='human')
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.expand_dims(next_state, axis=0)
                i += 1
                ep_rewards += reward
                # print(info)
                if done:
                    print("episode: {}/{}, steps: {}, score: {}".format(e, test_episodes, i, ep_rewards))
                    break

if __name__ == "__main__":
    env_name = 'gym_packman:Packman-v0'
    agent = DQNAgent(env_name)
    agent.run()
    agent.test(5)

# if __name__ == '__main__':
#     # run main loop for n secondes
#     num_envs = 5
#     epoce_time = 0.001 # hours
#     EPISODES = 5
#     env = make_mp_envs('gym_packman:Packman-v0', num_envs, 0)
    
#     for i in range(EPISODES):
#         env.reset() 
#         episodic_reward = 0
#         t_end = time.time() + 60 * 60 * epoce_time
#         while time.time() < t_end:
#             actions = [random.randrange(5) for _ in range(num_envs)]
#             # Recieve state and reward from environment.
#             _, rewards, dones, _ = env.step(actions)
#             episodic_reward += sum(rewards)/num_envs
#         # Reward of last episode
#         print("Episode * {} * Reward is ==> {}".format(i, episodic_reward))