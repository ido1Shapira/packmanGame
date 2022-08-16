# ignore warnings:
import os
from itertools import permutations 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import warnings
warnings.filterwarnings('ignore')
import gym
import numpy as np
import sys

STAY=0
UP=2
LEFT=1
DOWN=4
RIGHT=3

env_name = 'gym_packman:Packman-v0'
env = gym.make(env_name)

def min_solution(maze, x, y, x_dest, y_dest, path = None):
    def try_next(x, y):
        ' Next position we can try '
        return [(a, b) for a, b in [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)] if 0 <= a < n and 0 <= b < m]

    n = len(maze)
    m = len(maze[0])
    
    if path is None:
        path = [(x, y)]         # Init path to current position

    # Reached destionation
    if x == x_dest and y == y_dest:
        return path
    
    maze[x][y] = 0              # Mark current position so we won't use this cell in recursion
    
    # Recursively find shortest path
    shortest_path = None            
    for a, b in try_next(x, y):
        if maze[a][b]:
            last_path = min_solution(maze, a, b, x_dest, y_dest, path + [(a, b)])  # Solution going to a, b next
            
            if not shortest_path:
                shortest_path = last_path        # Use since haven't found a path yet
            elif last_path and len(last_path) < len(shortest_path):
                shortest_path = last_path       # Use since path is shorter
     
    
    maze[x][y] = 1           # Unmark so we can use this cell
    
    return shortest_path

def takeActionTo(current, to):
    if current[0] < to[0]:
        return DOWN #down
    elif current[0] > to[0]:
        return UP #up
    elif current[1] < to[1]:
        return RIGHT #right
    elif current[1] > to[1]:
        return LEFT #left
    return STAY


######################### Baselines ##############################################
def TSP():
    dirts_pos = env.get_dirts_position()
    computer_pos = env.get_computer_position()
    board = env.get_board()

    map_cost = {}
    for pos in dirts_pos:
        map_cost[pos] = {}

    perm = list(permutations(dirts_pos))
    min_cost = sys.maxsize
    dirts_order = []
    for path in perm:
        temp_cost = 0
        for i in range(len(path)-1):
            point1 = path[i]
            point2 = path[i+1]

            try:
                cost = map_cost[point1][point2]
            except KeyError:
                # inset cost
                d = len(min_solution(board, point1[0], point1[1], point2[0], point2[1]))
                map_cost[point1][point2] = d
                cost = map_cost[point1][point2]
            
            temp_cost += cost

        if(temp_cost < min_cost):
            min_cost = temp_cost
            dirts_order = path 
    
    optimal_path = min_solution(board, computer_pos[0], computer_pos[1], dirts_order[0][0], dirts_order[0][1])
    return takeActionTo(optimal_path[0], optimal_path[1])

def Closest():
    dirts_pos = env.get_dirts_position()
    computer_pos = env.get_computer_position()
    board = env.get_board()

    min_path = min_solution(board, computer_pos[0], computer_pos[1], dirts_pos[0][0], dirts_pos[0][1])
    for pos in dirts_pos:
        path = min_solution(board, computer_pos[0], computer_pos[1], pos[0],pos[1])
        if len(min_path) > len(path):
            min_path = path

    return takeActionTo(min_path[0], min_path[1])

def Farthest():
    dirts_pos = env.get_dirts_position()
    computer_pos = env.get_computer_position()
    human_pos = env.get_human_position()
    board = env.get_board()

    max_path = min_solution(board, computer_pos[0], computer_pos[1], dirts_pos[0][0], dirts_pos[0][1])
    for pos in dirts_pos:
        path_h = min_solution(board, human_pos[0], human_pos[1], pos[0],pos[1])
        path_c = min_solution(board, computer_pos[0], computer_pos[1], pos[0],pos[1])

        if len(path_c) <= len(path_h):
            if len(path_c) > len(max_path):
                max_path = path_c

    return takeActionTo(max_path[0], max_path[1])

    #         var d_h = SPA.getMinDistance(); //human distance
    #         SPA.run(player_pos, award_pos);
    #         var d_c = SPA.getMinDistance(); //computer distance
    #         if(d_c <= d_h) {
    #             //Check if the computer comes before the human
    #             if(d_c > max_d) {
    #                 max_d = d_c;
    #                 max_path = SPA.getSortestPath();
    #             }
    #         }
    #     }
    #     if(max_path == null) {
    #         return 32;
    #     }
    #     return this.takeActionTo(max_path[0], max_path[1]);

def Selfish():
    return 0

def Random():
    return env.get_random_valid_action('computer')


################################## MAIN ##################################

env.reset()

baselines = [
    'TSP', #: TSP(),
    'Closest', #: Closest(),
    'Farthest', #: Farthest(),
    'Selfish', #: Selfish(),
    'Random', #: Random()
]

# maps = ['map 3', 'map 4', 'map 5']

dir_map = 'map 5'
total_episodes = 500

print(dir_map + ' loaded')
for agent in baselines:
    print("Run agent: %s" % agent)
    scores = []
    for ep in range(1,total_episodes+1):
        env.reset() 
        episodic_reward = env.rewards['Start']
        while True:
            # time.sleep(0.1)
            # env.render()
            if agent == 'TSP':
                action = TSP()
            elif agent == 'Closest':
                action = Closest()
            elif agent == 'Farthest':
                action = Farthest()
            elif agent == 'Selfish':
                action = Selfish()
            elif agent == 'Random':
                action = Random()
            else:
                raise NotImplementedError()
            # Recieve state and reward from environment.
            next_state, reward, done, info = env.step(action)
            episodic_reward += reward

            # print(info)
            # End this episode when `done` is True
            if done:
                # print("episode: {}/{}, score: {}".format(ep, total_episodes, episodic_reward))
                scores.append(episodic_reward)
                break
    
    print(dir_map + ' :\n agent: ' + agent + ' average score is: ' + str(sum(scores) / len(scores)))
    scores = []