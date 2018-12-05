import numpy as np
from copy import *
from torch import optim
from torch import nn
import torch

class DDPG():
    def __init__(self, actor, critic, memory, obsspace_shape, \
                 actionspace_shape, action_range=(-1., 1.), normalize_returns=False, gamma=.99, batch_size=128, rho=.001):
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_critic = deepcopy(critic) #TODO: Deep copy might not work like the manual copying of the weights
        self.target_actor = deepcopy(actor)
        self.opts = {
            "actor" : optim.Adam(self.actor.parameters()),
            "critic": optim.Adam(self.critic.parameters()),
            "target_actor": optim.Adam(self.target_actor.parameters()),
            "target_critic": optim.Adam(self.target_critic.parameters())
        }
        self.rho = rho
        self.reward_scale = 1

    def pi(self, obs):
        """

        :param obs:
        :return: Returns max over actions given obs
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        action = self.actor(obs)
        action = torch.clamp(action, self.action_range[0], self.action_range[1])
        return action

    def target_Q(self, obs, rewards, terminals):
        return rewards + self.gamma * (1-terminals) * self.target_critic(obs, self.target_actor(obs))


    def train(self):
        def polyak_update(current, target):
            # Manually update target network model with polyak averaging
            cur_params = current.named_parameters()
            target_params = target.named_parameters()

            dict_target_params = dict(target_params)
            for c_name, c_param in cur_params:
                if c_name in dict_target_params:
                    dict_target_params[c_name].data.copy_((1. - self.rho)*dict_target_params[c_name].data + self.rho * c_param.data)

            target.load_state_dict(dict_target_params)

        batch = self.memory.sample(batch_size=self.batch_size)

        # TODO: normalize returns

        target_Q = self.target_Q(batch['obs1'], batch['rewards'], batch['terminals1'])

        # Zero all grads
        for opt in self.opts.values():
            opt.zero_grad()

        # Update actor and critic nets TODO: is it a problem that this is a sequential, not synced, update?
        mse = nn.MSELoss(reduction='elementwise_mean')
        critic_loss = mse(self.critic(batch['obs0'],batch['actions']),
            target_Q)
        critic_loss.backward()
        self.opts["critic"].step()

        actor_loss = -torch.sum(self.critic(batch["obs0"], self.actor(batch["obs0"])))/self.batch_size # TODO: make sure broadcasting isn't messing this up
        actor_loss.backward()
        self.opts["actor"].step()

        # Update target networks
        polyak_update(self.actor, self.target_actor)
        polyak_update(self.critic, self.target_critic)

        return critic_loss, actor_loss

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale
        self.memory.append(obs0, action, reward, obs1, terminal1)

    def reset(self):
        pass