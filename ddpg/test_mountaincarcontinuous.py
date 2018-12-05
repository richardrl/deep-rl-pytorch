import gym
from torch import nn

env=gym.make("MountainCarContinuous-v0")

class TestNet(nn.Module):
    def __init__():
        super(TestNet, self).__init__()

while True:
    obs, done = env.reset(), False
    while not done:
        env.render()
        env.step([1])