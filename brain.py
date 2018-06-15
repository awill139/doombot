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

class Brain:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()

doom_env = image_preprocess.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
num_actions = doom_env.action_space.n

cnn = CNN(num_actions)
softmax_body = SoftmaxBody(temp = 1.0)
brain = Brain(brain = cnn, body = softmax_body)

n_steps = exp_replay.NStepProg(doom_env, brain, 10)
memory = exp_replay.ReplayMem(n_steps = n_steps)

def elig_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cum_reward = 0.0 if series[-1].done else output[1].data.max
        for step in reversed(series[:-1]):
            cum_reward = cum_reward * gamma + step.reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cum_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)