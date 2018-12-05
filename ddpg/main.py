import torch
from ddpg import *
import argparse

import os

import gym
import torch
import torch.nn.functional as F

import torch.optim
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn as nn

import argparse

from datetime import datetime
from models import *
from memory import Memory


parser= argparse.ArgumentParser()
parser.add_argument('--exp_prefix', required=True, help="Prefix for experiment. Log files saved ./runs/<exp_prefix>...")
parser.add_argument('--lr_p', required=False, type=float, default=.01, help="Learning rate for policy optimizer. Default .01")
parser.add_argument('--lr_v', required=False, type=float, default=.1, help="Learning rate for value optimizer. Default .1")
parser.add_argument('--manual_seed', required=False, type=int, default=None, help="Manual random seed. Default None")
parser.add_argument('--num_runs', required=False, type=int, default=1, help="# runs. Default 1")

opt = parser.parse_args()
print(opt)

if opt.manual_seed:
    torch.manual_seed(opt.manual_seed)


def train(env, memory, actor, critic, render=False, n_rollout_steps=100, n_epoch_cycles=20, n_train_steps=50, writer=None):
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape)
    agent.reset()

    nepochs = 500
    obs, done, episode_reward = env.reset(), False, 0
    episode = 0
    for epoch in range(nepochs):
        for cycle in range(n_epoch_cycles):
            for t_rollout in range(n_rollout_steps):

                obs = obs.reshape((-1)) # Flatten obs each time
                action = agent.pi(obs).data.numpy()
                max_action = 1 # Scaling factor from default [-1,1] action range
                new_obs, r, done, info =env.step(max_action * action)
                if render:
                    env.render()
                episode_reward += r
                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs
                if done:
                    # agent.reset()
                    writer.add_scalar("critic loss ", critic_loss, episode)
                    writer.add_scalar("actor loss ", actor_loss, episode)
                    writer.add_scalar("episode reward ", episode_reward, episode)
                    print("critic loss " +str(critic_loss))
                    print("actor loss " + str(actor_loss))
                    print("episode reward "+ str(episode_reward))
                    print("\n")
                    obs, done, episode_reward = env.reset(), False, 0
                    episode+=1
            # Update models: after collecting n_rollout_steps of experience, update actor and critic
            for t_train in range(n_train_steps):
                critic_loss, actor_loss = agent.train()

def run(env_id, exp_prefix, lr_p, lr_v, run_num, seed=1, gamma=.97):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    try:
        os.makedirs(F"./runs/{exp_prefix}")
    except OSError:
        pass
    logdir = F"./runs/{exp_prefix}/lr_p{lr_p}-lr_v{lr_v}-run_num{run_num}-{current_time}"
    writer = SummaryWriter(logdir)

    env = gym.make(env_id)

    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(observation_shape=env.observation_space.shape)

    nactions = env.action_space.shape[-1]
    actor = Actor(nactions, observation_shape=env.observation_space.shape)

    torch.manual_seed(seed)
    train(env=env, memory=memory, actor=actor, critic=critic, writer=writer)

    writer.close()

if __name__ == "__main__":
    # Set seed
    # torch.manual_seed(1)
    import sys

    if opt.num_runs == 1:
        run("MountainCarContinuous-v0", opt.exp_prefix, opt.lr_p, opt.lr_v, 1)
    else:
        for run_num in range(opt.num_runs):
            run("MountainCarContinuous-v0", opt.exp_prefix, opt.lr_p, opt.lr_v, run_num)