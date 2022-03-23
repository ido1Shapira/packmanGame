# ignore warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import gym

env_name = 'gym_packman:Packman-v0'
env = gym.make(env_name)

total_episodes = 10

for ep in range(1,total_episodes+1):
    env.reset() 
    episodic_reward = 0
    while True:
        env.render()
        # action = env.get_random_valid_action('computer')
        action = 0
        # Recieve state and reward from environment.
        _, reward, done, _ = env.step(action)
        episodic_reward += reward
        # End this episode when `done` is True
        if done:
            break
    # Reward of last episode
    print("Episode * {} * Reward is ==> {}".format(ep, episodic_reward))