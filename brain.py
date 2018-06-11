import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
import exp_replay, image_preprocess

class CNN(nn.Module):
    def __init__(self, num_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = num_actions)
    
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = x.view(x.size(0), -1) # flatten x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SoftmaxBody(nn.Module):
    def __init__(self, temp):
        super(SoftmaxBody, self).__init__()
        self.temp = temp
    
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions