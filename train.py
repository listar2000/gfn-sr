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


def train_gfn_sr(batch_size, num_epochs, show_plot=False, use_gpu=True):
    torch.manual_seed(1234)
    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    print("training started with device", device)
    X = torch.empty(200, 1).uniform_(0, 1) * 5
    # y = X[:, 0] + 3
    y = X[:, 0] * 2
    action = Action(X.shape[1])
    env = SRTree(X, y, action_space=action, max_depth=2, loss="other")

    forward_policy = RNNForwardPolicy(batch_size, 250, env.num_actions, 1, model="lstm", device=device)
    backward_policy = CanonicalBackwardPolicy(env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2)

    flows, errs, avg_rewards = [], [], []

    for i in (p := tqdm(range(num_epochs))):
        s0 = env.get_initial_states(batch_size)
        s, log = model.sample_states(s0)
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 10 == 0:
            avg_reward = log.rewards.mean().item()
            p.set_description(f"{loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())
            avg_rewards.append(avg_reward)

    # codes for plotting loss & rewards
    if show_plot:
        train_plot(errs, flows)

    return model, env, errs


def run_torch_profile(prof_batch=32, prof_epochs=3, use_gpu=False):
    """
    Function for profiling the computation time of the model. See
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    for documentation and examples.
    """
    print("Starting torch profiling...")
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_training"):
            train_gfn_sr(prof_batch, prof_epochs, show_plot=False, use_gpu=use_gpu)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5000)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    model, env, errs = train_gfn_sr(batch_size, num_epochs, show_plot=True, use_gpu=True)
