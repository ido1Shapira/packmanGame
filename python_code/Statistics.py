#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import os

path = os.path.abspath('./data/packman-game-default-rtdb-export.json')
print(os.path.exists(path))

# In[3]:


# load data from the json file
with open(path) as train_file:
    data = json.load(train_file)

# In[4]:


participants_df = pd.DataFrame.from_dict(data['all-games'], orient='index')
# df.reset_index(level=0, inplace=True)
# df = df.dropna(subset=['log'])
participants_df

# In[5]:


participants_df.columns

# In[6]:


participants_df['gender'].value_counts()

# In[7]:


participants_df['education'].value_counts()

# In[ ]:


# # View some data

# In[8]:


raw_df_state_to_action = pd.DataFrame.from_dict(data['humanModel'], orient='index')
# df.reset_index(level=0, inplace=True)
# df = df.dropna(subset=['log'])
raw_df_state_to_action = raw_df_state_to_action.drop(0, axis=1)
raw_df_state_to_action


# In[9]:


class DisplayState:
    def __init__(self, state):
        self.size = len(state)
        self.h = self.size
        self.w = self.size
        self.raw_state = state
        self.board = np.array(state[0]).astype(np.float)
        self.human_trace = np.array(state[1]).astype(np.float)
        self.computer_trace = np.array(state[2]).astype(np.float)
        self.human_awards = np.array(state[3]).astype(np.float)
        self.computer_awards = np.array(state[4]).astype(np.float)
        self.all_awards = np.array(state[5]).astype(np.float)
        self.dict = {
            "Board": self.board,
            "Human trace": self.human_trace,
            "Computer trace": self.computer_trace,
            "Human awards": self.human_awards,
            "Computer awards": self.computer_awards,
            "All awards": self.all_awards,
        }

    def ToGrayScale(self, which='all'):
        if (which == 'all'):
            axes = []
            fig = plt.figure(figsize=(10, 8))
            i = 0
            j = 0
            for key in self.dict:
                axes.append(fig.add_subplot(2, 3, i + 1))
                i = i + 1
                subplot_title = ("Subplot: " + str(key))
                axes[-1].set_title(subplot_title)
                plt.imshow(self.dict[key])
            fig.tight_layout()
        else:
            plt.imshow(self.dict[which], interpolation='nearest')
        plt.show()

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def ToImage(self):
        try:
            i_a_h, j_a_h = np.where(self.human_awards == 1)  # indexes of the human_awards
        except:
            print("An exception occurred at: human_awards")
        try:
            i_a_c, j_a_c = np.where(self.computer_awards == 1)  # indexes of the computer_awards
        except:
            print("An exception occurred at: computer_awards")

        r = self.board / 10 + self.all_awards
        r += self.human_trace

        if (not np.any(self.all_awards)):
            g = np.zeros([10, 10])
        else:
            g = self.board + self.all_awards * 3

        b = self.board / 10 + self.all_awards / 10
        b += self.computer_trace

        if i_a_h.size != 0:
            r[i_a_h, j_a_h] += 0.5
            g[i_a_h, j_a_h] += 0.2
            b[i_a_h, j_a_h] += 0.2
        if i_a_c.size != 0:
            r[i_a_c, j_a_c] += 0.2
            g[i_a_c, j_a_c] += 0.2
            b[i_a_c, j_a_c] += 0.5

        r = self.NormalizeData(r)
        g = self.NormalizeData(g)
        b = self.NormalizeData(b)

        rgb = np.dstack((r, g, b))
        return rgb


# In[10]:


# def extractState(cell):
#     if cell != None:
#         ds = DisplayState(cell['state'])
#         return ds.ToImage()
#     return np.nan

def extractAction(cell):
    if cell != None:
        return int(cell['action'])
    return np.nan


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def extractState(cell):
    if cell == None:
        return cell
    board = np.array(cell['state'][0]).astype(float)
    human_trace = np.array(cell['state'][1]).astype(float)
    computer_trace = np.array(cell['state'][2]).astype(float)
    human_awards = np.array(cell['state'][3]).astype(float)
    computer_awards = np.array(cell['state'][4]).astype(float)
    all_awards = np.array(cell['state'][5]).astype(float)

    r = human_awards / 2 + human_trace
    g = board / 3 + all_awards
    b = computer_awards / 2 + computer_trace
    rgb = np.dstack((r, g, b))
    return NormalizeData(rgb)


state_df = pd.DataFrame(columns=raw_df_state_to_action.columns)
action_df = pd.DataFrame(columns=raw_df_state_to_action.columns)
for col in raw_df_state_to_action:
    state_df[col] = raw_df_state_to_action[col].apply(extractState)
    action_df[col] = raw_df_state_to_action[col].apply(extractAction)

# In[11]:


state_df

# In[12]:


action_df

# # Make Dataset

# In[13]:


for (idxRow, s1), (_, s2) in zip(state_df.iterrows(), action_df.iterrows()):
    for (idxCol, state), (_, action) in zip(s1.iteritems(), s2.iteritems()):
        if not np.isnan(action):
            im = Image.fromarray((state * 255).astype(np.uint8))
            path = f'data/humanModel/imagesDatabase/{int(action)}/{idxRow}_{idxCol}.png'
            print(f'{idxRow}_{idxCol}.png saved! at action {action}')
            im.save(path)
#         print (state, action, idxCol, idxRow)


# In[14]:


index = "-MhURGqHFd3RAOdztBo3"
col = 2
plt.imshow(state_df.loc[index, col])
title = "id: " + index + ", col: " + str(col) + ", action: " + str(action_df.loc[index, col])
plt.title(title)
plt.show()
