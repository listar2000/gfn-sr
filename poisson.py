import torch
from torch.nn.functional import one_hot
from torch.distributions.poisson import Poisson
from gflownet.env import Env


class PoissonPoint(Env):
    def __init__(self, mean=10):
        self.mean = mean
        self.num_actions = 2  # right or terminate
        self.state_dim = mean * 2  # terminate at 2x the mean
        self.poisson = Poisson(self.mean)
        # self.truncate_prob = 1 - self.poisson.cdf(self.state_dim)

    def update(self, s, actions):
        idx = s.argmax(1)
        right = actions == 0
        idx[right] = idx[right] + 1
        return one_hot(idx, self.state_dim).float()

    def mask(self, s):
        mask = torch.ones(len(s), self.num_actions)
        idx = s.argmax(1) + 1
        mask[idx % self.state_dim == 0, 0] = 0
        return mask

    def reward(self, s, truncate=False):
        """
        the reward should be proportional to the Poisson density
        """
        idx = s.argmax(1)
        log_probs = self.poisson.log_prob(idx)
        if truncate:
            is_zero = idx == 0
            log_probs[is_zero] = log_probs[is_zero] + self.truncate_prob
        return torch.exp(log_probs)
