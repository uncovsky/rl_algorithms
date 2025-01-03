import matplotlib.pyplot as plt
from matplotlib import cm

import gymnasium as gym
from math import sqrt
import numpy as np

class ToyTextWrapper:
    def __init__(self, env):
        self.env = env 
        
        # name of the env instance
        self.name = self.env.unwrapped.spec.id

        # stores transition info & rewards
        self.P = self.env.unwrapped.P


    def num_states(self):
        return self.env.observation_space.n

    def num_actions(self):
        return self.env.action_space.n


    def get_transition(self, state, act):
        return [ (succ, reward, prob, terminates ) for prob, succ, reward, terminates in self.P[state][act] ]

    def render(self):
        return self.env.render()
    
    def get_name(self):
        return self.name

    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()

    # get number of rows/columns in the rendered map
    def map_dimensions(self):
        height, width = 0, 0
        if "FrozenLake" in self.name:
            nS = self.num_states()
            height = int(sqrt(nS))
            width = height
        elif 'CliffWalking' in self.name:
            height = 4
            width = 12

        return height, width


    # translate action index to the actual movement direction
    def get_direction_mapping(self):
        action_dict = {}
        if "FrozenLake" in self.name:
            action_dict = {
                0 : "LEFT",
                1 : "DOWN",
                2 : "RIGHT",
                3 : "UP"
            }

        elif 'CliffWalking' in self.name:
            action_dict = {
                0 : "UP",
                1 : "RIGHT",
                2 : "DOWN",
                3 : "LEFT"
            }

        return action_dict
            




def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



# renders a heatmap of values in the environment
def render_heatmap(frame, values, rows, cols, env_name, algo_name, ax):

    cell_height = frame.shape[0] // rows
    values = np.array(values).reshape((rows, cols))

    if ax is None:
        ax = plt.gca()
    cmap = cm.get_cmap('Greys')

    im = ax.imshow(values, cmap=cmap, extent=[0, cols, 0, rows])
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")

    for i in range(rows):
        for j in range(cols):
            ax.text(j + 0.5, rows - i - 0.5, f'{values[i, j]:.2f}',
                    ha="center", va="center", color="red", fontsize=cell_height * 0.2)

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(1, values.shape[1]+1), minor=True)
    ax.set_yticks(np.arange(1, values.shape[0]+1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(f"Heatmap of {algo_name} value function on {env_name}.")
    # ax.set_aspect('auto')



# Show arrows from actions on env
"""
    Action directions contains movements of the agent (translated from action
    indices to LEFT/RIGHT/DOWN/UP)

    rows/cols describe the number of states in each dimension of the rendered
    map


"""
def render_actions(frame, movement_dirs, rows, cols, env_name, algo_name, ax):

    nS = len(movement_dirs)

    height, width, _ = frame.shape

    frame = rgb2gray(frame)

    env_image = ax.imshow(frame, cmap='gray')

    cell_height = height // rows
    cell_width = width // cols

    dir_to_arrow = {
        "LEFT": (-0.25 * cell_width, 0),   
        "DOWN": (0, 0.25 * cell_height),    
        "RIGHT": (0.25 * cell_width, 0),    
        "UP": (0, -0.25 * cell_height)     
    }

    for state in range(nS):

        row = state // cols
        col = state % cols

        # get arrow from action description
        direction = dir_to_arrow[movement_dirs[state]]
        
        center_x = col * cell_width + cell_width // 2
        center_y = row * cell_height + cell_height // 2

        dx, dy = direction
        
        # draw arrow in the current cell, pointing in the dir of optimal action
        ax.arrow(center_x, center_y, dx, dy, color='red', 
                  head_width=cell_width * 0.2, head_length=cell_height * 0.2)


    ax.set_title(f"{env_name} with {algo_name} policy actions")
    return ax


def visualize_policy(env, q_values, algo_name):

    # Render the environment without taking any steps
    env.reset()
    frame = env.render()

    # Get state value function
    state_values = [ np.max(qvals) for qvals in q_values ]

    r, c = env.map_dimensions()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Translate action indices to movement directions (the envs differ in the mapping)
    direction_map = env.get_direction_mapping()
    action_directions = [direction_map[np.argmax([q_vals])] for q_vals in q_values ]

    render_actions(frame, action_directions, r, c, env.get_name(), algo_name, ax1)
    render_heatmap(frame, state_values, r, c, env.get_name(), algo_name, ax2)

    plt.tight_layout()
    plt.show()

