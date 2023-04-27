import torch
from env import SRTree
from actions import Action


def prep_data_and_action():
    X = torch.randn((20, 2))
    y = X[:, 0] - X[:, 1]
    action = Action(2)
    return X, y, action


def test_env_reward_nrmse():
    X, y, action = prep_data_and_action()
    op_idx = action.action_names.index("-")
    # ground truth model
    ecd_a = torch.tensor([[op_idx, 1, 2]])
    tree = SRTree(X, y, action_space=action, max_depth=2)
    assert torch.abs(tree.reward(ecd_a) - 1.0) <= 1e-5
    # model with constant
    ecd_b = torch.tensor([[op_idx, 0, 2]])
    assert tree.reward(ecd_b) < 1.0


def test_env_reward_baseline():
    X, y, action = prep_data_and_action()
    op_idx = action.action_names.index("-")
    # ground truth model
    ecd_a = torch.tensor([[op_idx, 1, 2]])
    tree = SRTree(X, y, action_space=action, max_depth=2, loss="baseline")
    assert torch.abs(tree.reward(ecd_a) - 100.0) <= 1e-5

    ecd_b = torch.tensor([[op_idx, 1, 1]])
    # minimum reward for baseline loss
    assert torch.abs(tree.reward(ecd_b) - 1.0) <= 1e-5


def test_env_mask():
    X, y, action = prep_data_and_action()
    mul_idx, sin_idx = action.action_names.index("*"), action.action_names.index("sin")
    ecds = torch.tensor([[mul_idx, sin_idx, 1, -2, -1, -1, -1], [mul_idx, mul_idx, 1, 0, -2, -1, -1],
                         [mul_idx, mul_idx, sin_idx, -2, -2, -2, -1], [mul_idx, mul_idx, -2, -2, -2, -1, -1]])
    tree = SRTree(X, y, action_space=action, max_depth=3)
    print(tree.mask(ecds))