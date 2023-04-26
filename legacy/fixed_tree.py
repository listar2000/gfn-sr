from gflownet.env import Env
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from gflownet.gflownet import GFlowNet
from gflownet.utils import trajectory_balance_loss
from rnn_model import FixedTreeRNN
from torch.optim import Adam

NUM_FEATURES = 2
BIN_OPS = [
    torch.add,
    torch.mul,
    torch.sub,
]

UN_OPS = [
    torch.sin,
    torch.cos,
    torch.exp,
    torch.abs,
    lambda x: x
]


def evaluate(encoding, X):
    res = [0] * len(encoding)
    for i in reversed(encoding.nonzero().flatten()):
        s = int(encoding[i])
        if s <= NUM_FEATURES:
            res[i] = X[:, s - 1]
        elif s <= NUM_FEATURES + len(BIN_OPS):
            left_i = i * 2 + 1
            right_i = left_i + 1
            res[i] = BIN_OPS[s - NUM_FEATURES - 1](res[left_i], res[right_i])
        else:
            left_i = i * 2 + 1
            res[i] = UN_OPS[s - NUM_FEATURES - len(BIN_OPS) - 1](res[left_i])
    # print(res[0])
    return res[0]


class FixedTree(Env):
    def __init__(self, template, X: torch.Tensor, y: torch.Tensor):
        # plus 1 for termination action
        self.template = template.long()
        self.num_actions = NUM_FEATURES + len(BIN_OPS) + len(UN_OPS) + 1
        self.state_dim = len(template)
        self.X, self.y = X, y
        # params for the forward policy
        self.forward_params = {"template": self.template}

    def update(self, s: torch.Tensor, actions):
        if actions[0] == self.num_actions - 1:
            return s
        new_s = s.clone().detach()
        s0 = s[0, :] > 0
        idx = (self.template ^ s0).argmax()
        new_s[:, idx] = actions + 1
        return new_s

    def mask(self, s):
        mask = torch.zeros(self.num_actions)
        s0 = s[0, :] > 0
        idx = (self.template ^ s0).argmax()
        left_idx = idx * 2 + 1
        # termination case
        if not idx and s0[0]:
            mask[-1] = 1
        # leaf node case
        elif left_idx >= self.state_dim or not self.template[left_idx]:
            mask[:NUM_FEATURES] = 1
        # binary node case
        elif self.template[left_idx + 1]:
            mask[NUM_FEATURES:NUM_FEATURES+len(BIN_OPS)] = 1
        else:
            mask[NUM_FEATURES+len(BIN_OPS):-1] = 1
        return mask.expand(len(s), len(mask))

    def reward(self, s):
        # this might be a performance bottleneck
        # res has shape B x N where B is batch size and N is data size
        res = torch.zeros(len(s), len(self.X))
        for i in range(len(s)):
            res[i, ] = evaluate(s[i, ], self.X)
        mse = ((res - self.y.expand(len(s), len(self.y))) ** 2).mean(dim=1)
        # max_mse = ((self.y - self.y.mean()) ** 2).mean()
        # rewards = torch.clamp(100 * (1.0 - mse / max_mse), min=1)

        nrmse = torch.sqrt(mse) / torch.std(self.y)
        rewards = 1 / (1 + nrmse)
        return rewards


class FTForwardPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, s):
        return self.net(s)


class FTBackwardPolicy:
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

    def __call__(self, s):
        idx = s[0, :].nonzero().max()
        actions = s[:, idx].long() - 1
        probs = one_hot(actions, self.num_actions)
        return probs


def train_plot(errs, flows):
    fix, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    xs = range(1, len(errs) + 1)
    ax1.plot(xs, errs, "b", label="loss over time")
    ax1.set_xlabel("10 epochs")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(xs, flows, "b", label="total flow over time")
    ax2.set_xlabel("10 epochs")
    ax2.set_ylabel("total flow")
    ax2.legend()

    plt.show()


def train_fixed_tree(batch_size, num_epochs, show_plot=False, rnn=False):
    X = torch.randn(20, 2)
    y = (X[:, 1] + X[:, 0]) * X[:, 0]
    temp = torch.tensor([1, 1, 1, 1, 1, 1, 0])
    # X = torch.vstack([torch.empty(20).uniform_(-1, 1) for _ in range(3)]).T
    # y = (torch.sin(X[:, 1]) + X[:, 0]) * (torch.cos(X[:, 1]) + X[:, 2])
    # temp = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    env = FixedTree(temp, X, y)
    if rnn:
        forward_policy = FixedTreeRNN(2, 128, env.num_actions, template=temp)
    else:
        forward_policy = FTForwardPolicy(env.state_dim, 32, env.num_actions)
    backward_policy = FTBackwardPolicy(env.state_dim, num_actions=env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    opt = Adam(model.parameters(), lr=5e-3)

    flows, errs = [], []
    for i in (p := tqdm(range(num_epochs))):
        s0 = torch.zeros(batch_size, env.state_dim)
        s, log = model.sample_states(s0, return_log=True)
        log.back_probs.fill_(1.0)
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs,
                                       log.back_probs)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 50 == 0:
            p.set_description(f"{loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())

    # codes for plotting loss & rewards
    if show_plot:
        train_plot(errs, flows)

    return model, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    model, env = train_fixed_tree(batch_size, num_epochs, show_plot=True, rnn=True)