"""
CMPUT 652, Fall 2019 - Assignment #2

__author__ = "Craig Sherstan"
__copyright__ = "Copyright 2019"
__credits__ = ["Craig Sherstan"]
__email__ = "sherstan@ualberta.ca"
"""

import torch
from torch import nn
import numpy as np

# Fix network_factory
# Exploration? Epsilon or Random?

def network_factory(in_size, num_actions, env):
    """

    :param in_size:
    :param num_actions:
    :param env: The gym environment. You shouldn't need this, but it's included regardless.
    :return: A network derived from nn.Module
    """
    #network = nn.Sequential(nn.Linear(in_size, 16), nn.ReLU(), nn.Linear(16, num_actions), nn.Softmax(dim=-1))
    #return network
    pass
    
class PolicyNetwork(nn.Module):
    def __init__(self, in_size, num_actions, env):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
        self.action_space = np.arange(env.action_space.n)
        
    def forward(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


    def get_action(self, state):
        action_probs = self.network(torch.FloatTensor(state)).detach().numpy()
        action = np.random.choice(self.action_space, p=action_probs)
        return action

    #def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
    #    raise NotImplementedError



class ValueNetwork(nn.Module):
    def __init__(self, in_size, num_actions, env):
        super(ValueNetwork, self).__init__()
        self.network_v = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, 1))        
        
    def forward(self, state):
        value = self.network_v(torch.FloatTensor(state))
        return value
                

    #def get_action(self, inputs):
        """
        This function will be used to evaluate your policy.
        :param inputs: environmental inputs. These should be the environment observation wrapped in a tensor:
        torch.tensor(obs, device=device, dtype=torch.float32)
        :return: Should return a single integer specifying the action
        """
    #    raise NotImplementedError
