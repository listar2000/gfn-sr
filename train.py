import torch
import torch.autograd.profiler as profiler
import argparse
import matplotlib.pyplot as plt
from env import SRTree
from actions import Action
from policy import RNNForwardPolicy, CanonicalBackwardPolicy
from gflownet import GFlowNet, trajectory_balance_loss
from tqdm import tqdm


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


def train_gfn_sr(batch_size, num_epochs, show_plot=False):
    torch.manual_seed(1234)
    X = torch.empty(200, 1).uniform_(0, 1) * 5
    # y = (X[:, 1] + X[:, 2]) * torch.exp(X[:, 0]) + torch.randn(200) * 0.1
    # y = X[:, 0] + X[:, 0] ** 2 + X[:, 0] ** 3 + X[:, 0] ** 4 + X[:, 0] ** 5
    # y = torch.log(X[:, 0] + 1) + torch.log(X[:, 0] ** 2 + 1)
    # y = torch.sqrt(X[:, 0])
    y = X[:, 0] ** 5 + X[:, 0] ** 4 + X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0]
    action = Action(X.shape[1])
    env = SRTree(X, y, action_space=action, max_depth=5, loss="other")

    forward_policy = RNNForwardPolicy(batch_size, 500, env.num_actions, model="gru")
    backward_policy = CanonicalBackwardPolicy(env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    flows, errs = [], []
    for i in (p := tqdm(range(num_epochs))):
        s0 = env.get_initial_states(batch_size)
        s, log = model.sample_states(s0, return_log=True)
        if not torch.isfinite(log.rewards).all():
            print(log.rewards)
            assert False
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs,
                                       log.back_probs)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 10 == 0:
            p.set_description(f"{loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())

    # codes for plotting loss & rewards
    if show_plot:
        train_plot(errs, flows)

    return model, env, errs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    model, env, errs = train_gfn_sr(batch_size, num_epochs, show_plot=True)
