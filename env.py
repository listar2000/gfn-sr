from gflownet.env import Env
from actions import Action, ExpressionTree
import torch


class SRTree(Env):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, action_space: Action,
                 max_depth=4, placeholder=-2, loss="nrmse", loss_thres=1, inner_loop_config=None):
        # plus 1 for termination action
        self.action_space = action_space
        self.action_fns = action_space.action_fns
        self.action_names = action_space.action_names
        self.action_arities = torch.tensor(action_space.action_arities)
        self.num_actions = len(self.action_fns) + 1
        self.state_dim = 2 ** max_depth - 1
        self.placeholder = placeholder
        self.loss = loss
        self.loss_thres = loss_thres

        # training configs for the inner loop
        if not inner_loop_config:
            inner_loop_config = {
                "optim": "rmsprop",
                "iteration": 10,
                "lr": 0.01,
                "loss": "mse",
            }
        self.inner_loop_config = inner_loop_config
        self.inner_eval_config = inner_loop_config.copy()
        self.inner_eval_config["iteration"] = 100

        self.best_reward = -torch.inf
        self.best_expr = None

        self.X, self.y = X, y

    def get_initial_states(self, batch_size=16):
        init_states = -1 * torch.ones((batch_size, self.state_dim)).long()
        init_states[:, 0] = self.placeholder
        return init_states

    def update(self, encodings: torch.Tensor, actions: torch.Tensor):
        new_encodings = encodings.clone()
        indices = (new_encodings == self.placeholder).long().argmax(axis=1)
        new_encodings[torch.arange(len(encodings)), indices] = actions
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
        mask[has_left_sib, :] = sub_mask
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
            final_reward = expression.optimize_constant(self.X, self.y, self.inner_loop_config)
            if torch.isnan(final_reward) or torch.isinf(final_reward):
                loss[i] = torch.inf
            else:
                loss[i] = final_reward

        if self.loss == "nrmse":
            nrmse = torch.sqrt(loss) / torch.std(self.y)
            rewards = torch.clamp(self.loss_thres / (self.loss_thres + nrmse), min=0.01)
        else:
            # TODO: alpha1 * fitting loss + alpha2 * structure loss
            max_mse = ((self.y - self.y.mean()) ** 2).mean()
            rewards = torch.clamp(100 * (1.0 - loss / max_mse), min=0.01)

        if len(rewards) > 1 and torch.max(rewards) > self.best_reward:
            best_reward_vanilla = torch.max(rewards)
            best_action = torch.argmax(rewards)
            best_expr = ExpressionTree(encodings[best_action], self.action_fns, self.action_arities, self.action_names)
            best_reward_optimized = best_expr.optimize_constant(self.X, self.y, self.inner_eval_config)
            print(f"\nnew best reward (vanilla): {best_reward_vanilla}")
            print(f"mse (optimized): {best_reward_optimized}")
            print(f"expr: {str(best_expr)}")
            self.best_reward = best_reward_vanilla
            self.best_expr = best_expr
        return rewards
