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
from network import network_factory
from network import PolicyNetwork
from network import ValueNetwork
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
    

def discount_rewards(rewards, gamma=1.0):
    # Discounts rewards and stores their cumulative return in reverse

    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r


def reinforce(env, policy_estimator, value_estimator, num_episodes, # value_estimator=None,
              batch_size=1, gamma=1.0):

    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 0
    writer = SummaryWriter()
    
    # Define optimizer
    optimizer =   optim.Adam(policy_estimator.network.parameters(),  lr=0.0025)
    optimizer_v = optim.Adam(value_estimator.network_v.parameters(), lr=0.001)
    
    action_space = np.arange(env.action_space.n)
    flag = 1     # 1 for train, 0 for test
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        
        while complete == False:
        
            # Gets reward and next state
            
            action = policy_estimator.get_action(s_0)
            s_1, r, complete, _ = env.step(action)
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # Checks if episode is over
                
            if complete:
                
                batch_counter += 1
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                
                total_rewards.append(sum(rewards))
                
                # Updates after batch of episodes, here batch is 1
                
                if batch_counter == batch_size:
                    if flag == 1:
                    
                        # Value update
                        
                        state_tensor_v = torch.tensor(batch_states, dtype=torch.float32)
                        reward_tensor_v = torch.tensor(batch_rewards, dtype=torch.float32)
                        value_estimates = value_estimator.forward(state_tensor_v)
                        loss_v = torch.mean((reward_tensor_v-value_estimates)**2)
                        loss_v.backward(retain_graph=True)
                        optimizer_v.step()                    
                        
                        # Policy update
                        
                        state_tensor = torch.tensor(batch_states, dtype=torch.float32)
                        reward_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                        action_tensor = torch.tensor(batch_actions, dtype=torch.int32)
                        loss = - torch.mean(policy_estimator.forward(state_tensor).log_prob(action_tensor)*(reward_tensor-value_estimates)) #-value_estimates
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 0
                    
                print("Ep: {} Average of last 100: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-100:])))
                
                # Saves policy 
                
                if (ep + 1) % 10000 == 0:
                    torch.save(pe.network.state_dict(), 'saved_network_'+ str(ep + 1) + '_baseline.pkl')
                
                if flag == 1:
                    
                    # Tensorboard plots
                    
                    writer.add_scalar('return', total_rewards[-1], ep) #discounted rewards with gamma = 1, hence undiscounted
                    writer.add_scalar('loss/policy', loss, ep)
                    writer.add_scalar('loss/value', loss_v, ep)
                    writer.add_scalar('loss/total', 0.98*loss + 0.02*loss_v, ep)
                    for name, params in zip(policy_estimator.network.state_dict().keys(), policy_estimator.network.parameters()):
                        average_grad = torch.mean(params.grad**2)
                        writer.add_scalar('gradient_policy/'+str(name), average_grad, ep)
                    for name, params in zip(value_estimator.network_v.state_dict().keys(), value_estimator.network_v.parameters()):
                        average_grad = torch.mean(params.grad**2)
                        writer.add_scalar('gradient_value/'+str(name), average_grad, ep)
                    
                
    return total_rewards

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numrun = 1
    
    for run in range(numrun):
        env = make_env()

        in_size = env.observation_space.shape[0]
        num_actions = env.action_space.n

        network = network_factory(in_size, num_actions, env)
        network.to(device)
        pe = PolicyNetwork(network)
        
        # Load policy to test
        #pe.network.load_state_dict(torch.load('saved_network_50000_baseline.pkl'))
        
        ve = ValueNetwork(in_size)
        ep_returns = reinforce(env, pe, ve, episodes) #,ve , loss_policy, loss_value
            
        #fwrite = open('runs_data/'+str(run)+'.pkl','wb')
        #fwrite = open('runs_data/0.pkl','wb')
        #pickle.dump(ep_returns, fwrite)
        #fwrite.close()
            
        
        
    window = 100
    plt.figure(figsize=(12,8))
    plt.plot(sliding_window(ep_returns, window))
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return (Sliding Window 100)")
    plt.show()

    # save your network
    #torch.save(pe.network.state_dict(), 'saved_network_50000_baseline.pkl')
    
