from gflownet.env import Env
from actions import Action, ExpressionTree
import torch


class SRTree(Env):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, action_space: Action,
                 max_depth=4, placeholder=-2, loss="nrmse", inner_loop_config=None):
        # plus 1 for termination action
        self.action_space = action_space
        self.action_fns = action_space.action_fns
        self.action_names = action_space.action_names
        self.action_arities = action_space.action_arities
        self.num_actions = len(self.action_fns) + 1
        self.state_dim = 2 ** max_depth - 1
        self.placeholder = placeholder
        self.loss = loss

        # training configs for the inner loop
        if not inner_loop_config:
            inner_loop_config = {
                "optim": "lbfgs",
                "iteration": 100,
                "loss": "mse",
            }
        self.inner_loop_config = inner_loop_config

        self.X, self.y = X, y

    def update(self, encodings: torch.Tensor, actions: torch.Tensor):
        new_encodings = encodings.clone()
        indices = (new_encodings == self.placeholder).argmax(axis=1)
        new_encodings[:, indices] = actions

        # setting new children locations to be placeholder
        new_action_arities = self.action_arities[actions]
        left_idx = indices * 2 + 1
        right_idx = left_idx + 1

        is_unary = new_action_arities == 1
        new_encodings[is_unary, left_idx[is_unary]] = self.placeholder
        is_binary = new_action_arities == 2
        new_encodings[is_binary, left_idx[is_binary]] = self.placeholder
        new_encodings[is_binary, right_idx[is_binary]] = self.placeholder
        return new_encodings

    def mask(self, encodings: torch.Tensor):
        M = len(encodings)
        ecd_mask = (encodings == self.placeholder)
        indices = ecd_mask.long().argmax(axis=1)
        # identify complete trees
        done_idx = (ecd_mask.sum(axis=1) == 0)

        mask = torch.zeros((M, self.num_actions))
        mask[done_idx, -1] = 1
        mask[~done_idx, :-1] = 1

        left_idx = indices * 2 + 1
        # RULE 1: check whether the most recent token is leaf or not
        is_leaf = left_idx >= encodings.shape[1]
        mask[is_leaf, self.action_space.feat_num:-1] = 0
        # RULE 2: check whether the parent is unary, in which case we disallow constant
        par_idx = torch.div(indices - 1, 2, rounding_mode='floor').clamp(min=0)
        min_action_idx = self.action_space.feat_num + self.action_space.op_num
        is_par_unary = encodings[torch.arange(M), par_idx] < min_action_idx
        mask[is_par_unary, 0] = 0
        # RULE 3: left sibling cannot be constant already
        has_left_sib = (indices % 2 == 0) & (indices > 0)
        sub_mask = mask[has_left_sib, :]
        left_constant_idx = encodings[has_left_sib, indices[has_left_sib] - 1] == 0
        sub_mask[left_constant_idx, 0] = 0
        return mask

    def reward(self, encodings):
        """
        ! we assume that `encodings` here are valid (complete) trees
        """
        N = len(encodings)
        loss = torch.zeros(N)
        for i in range(N):
            expression = ExpressionTree(encodings[i], self.action_fns, self.action_arities, self.action_names)
            # perform inner optimization
            loss[i] = expression.optimize_constant(self.X, self.y, self.inner_loop_config)

        if self.loss == "nrmse":
            nrmse = torch.sqrt(loss) / torch.std(self.y)
            rewards = 1 / (1 + nrmse)
        else:
            max_mse = ((self.y - self.y.mean()) ** 2).mean()
            rewards = torch.clamp(100 * (1.0 - loss / max_mse), min=1)
        return rewards


if __name__ == '__main__':
    def test_reward():
        X = torch.linspace(0, 1, 100)
        y = torch.pi * X
        X = X.unsqueeze(1)
        action = Action(1)
        ecd = torch.tensor([[8, 0, 1]])
        tree = SRTree(X, y, action_space=action, max_depth=2)
        tree.reward(ecd)

    def test_mask():
        ecds = torch.tensor([[9, 3, 1, -2, -1, -1, -1], [9, 9, 1, 1, -2, -1, -1],
                             [9, 10, 3, -2, -2, -2, -1], [9, 10, -2, -2, -2, -1, -1]])
        action = Action(1)
        tree = SRTree(None, None, action_space=action, max_depth=3)
        print(tree.mask(ecds))

    test_mask()