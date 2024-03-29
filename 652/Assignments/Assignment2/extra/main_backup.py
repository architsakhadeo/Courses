#!/usr/bin/env python3

"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

"""
You are free to additional imports as needed... except please do not add any additional packages or dependencies to
your virtualenv other than those specified in requirements.txt. If I can't run it using the virtualenv I specified,
without any additional installs, there will be a penalty.

I've included a number of imports that I think you'll need.
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import gym
from network_backup import network_factory
from network_backup import PolicyNetwork
from network_backup import ValueNetwork
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import os
import sys
from torch import nn
from torch import optim
import pickle


# prevents type-3 fonts, which some conferences disallow.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_env():
    env = gym.make('CartPole-v0')
    return env


def sliding_window(data, N):
    """
    For each index, k, in data we average over the window from k-N-1 to k. The beginning handles incomplete buffers,
    that is it only takes the average over what has actually been seen.
    :param data: A numpy array, length M
    :param N: The length of the sliding window.
    :return: A numpy array, length M, containing smoothed averaging.
    """

    idx = 0
    window = np.zeros(N)
    smoothed = np.zeros(len(data))

    for i in range(len(data)):
        window[idx] = data[i]
        idx += 1

        smoothed[i] = window[0:idx].mean()

        if idx == N:
            window[0:-1] = window[1:]
            idx = N - 1

    return smoothed
    

def discount_rewards(rewards, gamma=1):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r


def reinforce(env, policy_estimator, value_estimator, num_episodes, # value_estimator=None,
              batch_size=1, gamma=1):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 0
    #loss_policy = []
    #loss_value = []
    
    
    # Define optimizer
    optimizer =   optim.Adam(policy_estimator.network.parameters(),  lr=0.0025)
    optimizer_v = optim.Adam(value_estimator.network_v.parameters(), lr=0.001)
    
    action_space = np.arange(env.action_space.n)
    flag = 0 # 1 for train, 0 for test
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            #action_probs = policy_estimator.forward(s_0).detach().numpy()
            action = policy_estimator.get_action(s_0)#np.random.choice(action_space, p=action_probs)
            s_1, r, complete, _ = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If complete, batch data
            if complete:
                
                batch_counter += 1
                batch_rewards.extend(discount_rewards(rewards, gamma))
                #batch_rewards -= value_estimates.tolist()
                batch_states.extend(states)
                batch_actions.extend(actions)
                
                total_rewards.append(sum(rewards))
                
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    if flag == 1:
                        state_tensor_v = torch.FloatTensor(batch_states)
                        reward_tensor_v = torch.FloatTensor(batch_rewards)
                        value_estimates = value_estimator.forward(state_tensor_v)
                        # Actions are used as indices, must be LongTensor
                                            
                        # Calculate loss
                        loss_v = torch.mean((reward_tensor_v-value_estimates)**2)
                        #loss_value.append(loss_v)
                        # Calculate gradients
                        optimizer_v.zero_grad()
                        loss_v.backward(retain_graph=True)
                         #Apply gradients
                        optimizer_v.step()                    
                        

                        
                        
                        state_tensor = torch.FloatTensor(batch_states)
                        reward_tensor = torch.FloatTensor(batch_rewards)
                        # Actions are used as indices, must be LongTensor
                        action_tensor = torch.LongTensor(batch_actions)
                        
                        # Calculate loss
                        loss = -torch.mean(torch.log(policy_estimator.forward(state_tensor))[np.arange(len(action_tensor)), action_tensor]*(reward_tensor)) #-value_estimates
                        #loss_policy.append(loss)
                        # Calculate gradients
                        optimizer.zero_grad()
                        loss.backward()
                        # Apply gradients
                        optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 0
                    
                # Print running average
                print("Ep: {} Average of last 100: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-100:])))
                
    return total_rewards#, np.array(loss_policy), np.array(loss_value)


if __name__ == '__main__':

    """
    You are free to add additional command line arguments, but please ensure that the script will still run with:
    python main.py --episodes 10000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", default=10000, type=int, help="Number of episodes to train for")
    args = parser.parse_args()

    episodes = args.episodes

    """
    It is unlikely that the GPU will help in this instance (since the size of individual operations is small) - in fact 
    there's a good chance it could slow things down because we have to move data back and forth between CPU and GPU.
    Regardless I'm leaving this in here. For those of you with GPUs this will mean that you will need to move your 
    tensors to GPU.
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numrun = 1
    
    for run in range(numrun):
        env = make_env()

        in_size = env.observation_space.shape[0]
        num_actions = env.action_space.n

        #pe = policy_estimator(env)
        pe = PolicyNetwork(in_size, num_actions, env)
        pe.network.load_state_dict(torch.load('saved_network_50k_as.pkl'))
        ve = ValueNetwork(in_size, num_actions, env)
        ep_returns = reinforce(env, pe, ve, episodes) #,ve , loss_policy, loss_value
        
        #loss_total = loss_policy + loss_value
        #fwrite = open('runs_data/'+str(run)+'.pkl','wb')
        #fwrite = open('runs_data/29.pkl','wb')
        #pickle.dump(ep_returns, fwrite)
        #fwrite.close()
        
        #writer = SummaryWriter()
    window = 100
    plt.figure(figsize=(12,8))
    plt.plot(sliding_window(ep_returns, window))
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (Sliding Window 100)")
    plt.show()

    # TODO: save your network
    #torch.save(pe.network.state_dict(), 'saved_network_50k_as.pkl')
    
