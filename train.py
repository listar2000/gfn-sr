import torch
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
    X = torch.randn(20, 2)
    y = (X[:, 1] + X[:, 0]) * X[:, 0]
    action = Action(X.shape[1])
    env = SRTree(X, y, action_space=action, max_depth=3)

    forward_policy = RNNForwardPolicy(batch_size, 128, env.num_actions)
    backward_policy = CanonicalBackwardPolicy(env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    flows, errs = [], []
    for i in (p := tqdm(range(num_epochs))):
        s0 = env.get_initial_states(batch_size)
        s, log = model.sample_states(s0, return_log=True)
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs,
                                       log.back_probs)
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    model, env = train_gfn_sr(batch_size, num_epochs, show_plot=True)