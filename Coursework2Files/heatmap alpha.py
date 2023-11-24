pip install seaborn

from utils import DQN, ReplayBuffer, greedy_action, epsilon_greedy, update_target, loss

import torch # ML library 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

import gym # provides various RL environments
import matplotlib.pyplot as plt

import os
import fnmatch
import logging 
import seaborn as sns

# Parameters 
NUM_RUNS = 3
# EPSILON = 0.1     # placeholder value - you should implement your own exploration schedule
num_episodes = 300
batch_size = 128
buffer_size = 100000
update_period = 15
initial_epsilon = 1
initial_alpha = 1
min_epsilon = 0.1
decay_rate = 0.98


env = gym.make('CartPole-v1')

# Assuming runs_results is defined
runs_results = []

decays = [0.99, 0.98, 0.955, 0.94]
minima = [0.1, 0.01, 0.001]

# Create a matrix to store the mean episode durations for different configurations
mean_durations = np.zeros((len(decays), len(minima)))

for i, decay in enumerate(decays):
    for j, min_alpha in enumerate(minima):
        # Assuming runs_results contains lists of episode durations
        # Calculate mean of the episode durations for each configuration of buffer_size and batch_size
        for run in range(NUM_RUNS):
            print(f"Starting run {run+1} of {NUM_RUNS}")

            architecture = [4, 32, 32, 2]
            policy_net = DQN(architecture) # input = 4 and output = 2 (0 or 1) 
            target_net = DQN(architecture)
            update_target(target_net, policy_net)
            target_net.eval()
            
            # Layer size
            layer_size = architecture[1]
    
            # optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
            memory = ReplayBuffer(buffer_size) # 1 originally 

            steps_done = 0

            episode_durations = []

            # EPSILON = 0.2
            EPSILON = initial_epsilon
            lr = initial_alpha

            for i_episode in range(num_episodes): # can adjust num_episodes 
                lr = max(min_alpha, lr*decay)
                optimizer = optim.SGD(policy_net.parameters(), lr)

                EPSILON = max(min_epsilon, EPSILON*decay_rate)
                if (i_episode+1) % 50 == 0:
                    print("episode ", i_episode+1, "/", 300)

                observation, info = env.reset()
                state = torch.tensor(observation).float()

                done = False
                terminated = False
                t = 0
                while not (done or terminated):

                    # Select and perform an action
                    action = epsilon_greedy(EPSILON, policy_net, state)
                    # action = greedy_action(policy_net, state) # can try without epsilon greedy 

                    # reward is a scalar, [reward] creates a one-element list, which is converted into a tensor
                    observation, reward, done, terminated, info = env.step(action)
                    reward = torch.tensor([reward])
                    action = torch.tensor([action])
                    next_state = torch.tensor(observation).reshape(-1).float()

                    # store in replay buffer
                    memory.push([state, action, next_state, reward, torch.tensor([done])])

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    if not len(memory.buffer) < batch_size: # only proceed if enough data in buffer 
                        transitions = memory.sample(batch_size) # sample one transition from buffer
                        state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions)) # and unpack
                        # Compute loss
                        mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                        # Optimize the model
                        optimizer.zero_grad() # from optimizer, reset gradient to prep for new step 
                        mse_loss.backward() # store gradient of the loss wrt each para 
                        optimizer.step() # carry out gradient dedscent step & update of para 
                    
                    if done or terminated:
                        episode_durations.append(t + 1)
                    t += 1
                # Update the target network, copying all weights and biases in DQN
                if i_episode % update_period == 0: 
                    update_target(target_net, policy_net)
            runs_results.append(episode_durations)
        print('Complete')

        mean_duration = np.mean([np.mean(run[150:300]) for run in runs_results])
        mean_durations[i][j] = mean_duration

# Create subplots
fig, ax = plt.subplots(figsize=(8, 6))

# Create a heatmap using Seaborn on Matplotlib
sns.heatmap(mean_durations, annot=True, fmt=".2f", cmap='viridis', xticklabels=minima, yticklabels=decays, ax=ax)

# Setting labels, ticks, and title 
ax.set_title('Effect of varying the decay rate and minimum bound of α: mean return over 3 runs') 
ax.set_xlabel('Minimum bound of α')
ax.set_ylabel('Decay rate of α')

# Get the current Axes instance on the current figure
cax = plt.gcf().axes[-1]

# Create a colorbar for the entire figure
cbar = plt.colorbar(ax.collections[0], cax=cax)
cbar.set_label('Mean Return')

plt.tight_layout()
plt.show()