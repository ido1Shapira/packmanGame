from gym.envs.registration import register

register(
    id='Packman-v0',
    entry_point='gym_packman.envs:PackmanEnv',
    max_episode_steps=200,
)