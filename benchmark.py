import json

import torch
from tqdm import tqdm

from actions import Action
from env import SRTree
from gflownet import trajectory_balance_loss, GFlowNet
from policy import RNNForwardPolicy, CanonicalBackwardPolicy

SEED = 2023


def generate_x(num: int, low: int = 0, high: int = 1, dim: int = 1):
    torch.manual_seed(2023)
    return torch.empty(num, dim).uniform_(low, high)


def nguyen_1(num: int = 50):
    """
    x^3 + x^2 + x
    """
    xs = generate_x(num, -1, 1)
    ys = xs[:, 0] ** 3 + xs[:, 0] ** 2 + xs[:, 0]
    return xs, ys, 4


def nguyen_3(num: int = 50):
    """
    x^5 + x^4 + x^3 + x^2 + x
    """
    xs = generate_x(num, -1, 1)
    ys = xs[:, 0] ** 5 + xs[:, 0] ** 4 + xs[:, 0] ** 3 + xs[:, 0] ** 2 + xs[:, 0]
    return xs, ys, 5


def nguyen_5(num: int = 50):
    """
    sin(x^2) * cos(x) - 1
    """
    xs = generate_x(num, -10, 0)
    ys = torch.sin(xs[:, 0] ** 2) * torch.cos(xs[:, 0]) - 1
    return xs, ys, 5


def nguyen_6(num: int = 50):
    """
    sin(x) + sin(x + x^2)
    """
    xs = generate_x(num, -5, 5)
    ys = torch.sin(xs[:, 0]) + torch.sin(xs[:, 0] + xs[:, 0] ** 2)
    return xs, ys, 4


def nguyen_7(num: int = 50):
    """
    log(x + 1) + log(x^2 + 1)
    """
    xs = generate_x(num, -1, 1)
    ys = torch.log(xs[:, 0] + 1) + torch.log(xs[:, 0] ** 2 + 1)
    return xs, ys, 4


def nguyen_8(num: int = 50):
    """
    sqrt(x)
    """
    xs = generate_x(num, 0, 1)
    ys = torch.sqrt(xs[:, 0])
    return xs, ys, 3


def nguyen_9(num: int = 50):
    """
    2 * sin(x0) * cos(x1)
    """
    xs = generate_x(num, -5, 5, dim=2)
    ys = 2 * torch.sin(xs[:, 0]) * torch.cos(xs[:, 1])
    return xs, ys, 4


def nguyen_12(num: int = 50):
    """
    x0^3 - 0.5 * x1^2
    """
    xs = generate_x(num, -5, 5, dim=2)
    ys = xs[:, 0] ** 3 - 0.5 * xs[:, 1]
    return xs, ys, 4


NGUYEN_TESTS = [
    nguyen_1, nguyen_3, nguyen_5, nguyen_6, nguyen_7, nguyen_8,
    nguyen_9, nguyen_12
]


def progressive_depth_algo(test, idx, begin_depth=2, end_depth=5):
    batch_size = 256
    num_epochs = 1000

    print(f"start benchmarking test {idx}")
    xs, ys, _ = test()

    action = Action(xs.shape[1])

    for depth in range(begin_depth, end_depth + 1):

        print("progressive algo: depth {}".format(depth))
        env = SRTree(xs, ys, action_space=action, max_depth=depth, loss="dynamic", update_interval=1000)

        forward_policy = RNNForwardPolicy(batch_size, 250, env.num_actions, num_layers=2, model="lstm")
        backward_policy = CanonicalBackwardPolicy(env.num_actions)
        model = GFlowNet(forward_policy, backward_policy, env)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for i in (p := tqdm(range(num_epochs))):
            s0 = env.get_initial_states(batch_size)
            s, log = model.sample_states(s0)
            # if not torch.isfinite(log.rewards).all():
            #     print(log.rewards)
            #     assert False
            loss = trajectory_balance_loss(log.total_flow,
                                           log.rewards,
                                           log.fwd_probs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 10 == 0:
                p.set_description(f"{loss.item():.3f}")

        test_log = {
            "BEST_REWARD": env.reward_manager.best_reward.item(),
            "BEST_EXPR": str(env.reward_manager.best_expr)
        }

        print(test_log)


if __name__ == "__main__":
    # batch_size = 256
    # num_epochs = 500
    # json_path = "./benchmark/lstm_bs_256_update_1000_gamma_099_q_09.json"
    # test_log = []
    # for idx, test in enumerate(NGUYEN_TESTS):
    #     print(f"start benchmarking test {idx}")
    #     xs, ys, depth = test()
    #     print(xs.shape, ys.shape)
    #
    #     action = Action(xs.shape[1])
    #     env = SRTree(xs, ys, action_space=action, max_depth=depth, loss="dynamic", update_interval=1000)
    #
    #     forward_policy = RNNForwardPolicy(batch_size, 250, env.num_actions, num_layers=2, model="lstm")
    #     backward_policy = CanonicalBackwardPolicy(env.num_actions)
    #     model = GFlowNet(forward_policy, backward_policy, env)
    #     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    #
    #     for i in (p := tqdm(range(num_epochs))):
    #         s0 = env.get_initial_states(batch_size)
    #         s, log = model.sample_states(s0)
    #         # if not torch.isfinite(log.rewards).all():
    #         #     print(log.rewards)
    #         #     assert False
    #         loss = trajectory_balance_loss(log.total_flow,
    #                                        log.rewards,
    #                                        log.fwd_probs)
    #         loss.backward()
    #         opt.step()
    #         opt.zero_grad()
    #         if i % 10 == 0:
    #             p.set_description(f"{loss.item():.3f}")
    #
    #     test_log.append({
    #         "TEST_INDEX": idx,
    #         "BEST_REWARD": env.reward_manager.best_reward.item(),
    #         "BEST_EXPR": str(env.reward_manager.best_expr)
    #     })
    #
    #     print(test_log)
    #     with open(json_path, "w") as f:
    #         json.dump(test_log, f, indent=4)
    progressive_depth_algo(nguyen_5, 3, begin_depth=2, end_depth=3)
