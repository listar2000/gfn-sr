from gflownet.env import Env
from actions import Action, ExpressionTree, optimize_constant, ETEnsemble
from gflownet.utils import LossBuffer
import torch.nn as nn
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
                "iteration": 15,
                "lr": 0.01,
                "loss": "mse",
            }
        self.inner_loop_config = inner_loop_config
        self.inner_eval_config = inner_loop_config.copy()
        self.inner_eval_config.update({'iteration': 100, 'optim': 'lbfgs'})

        self.loss_buffer = LossBuffer(update_interval=1000)
        self.criterion = nn.MSELoss()
        self.reward_manager = None

        self.X, self.y = X, y

    def set_up_reward_manager(self):
        assert self.reward_manager is None
        from reward import TSSReward, NRMSEReward, DynamicTSSReward, StructureReward
        if self.loss == 'nrmse':
            self.reward_manager = NRMSEReward(self, verbose=True)
        elif self.loss == 'dynamic':
            self.reward_manager = DynamicTSSReward(self, verbose=True)
        elif self.loss == 'struct':
            self.reward_manager = StructureReward(self, verbose=True)
        else:
            self.reward_manager = TSSReward(self, verbose=True)

    def get_initial_states(self, batch_size=16):
        init_states = -1 * torch.ones((batch_size, self.state_dim), dtype=torch.long)
        init_states[:, 0] = self.placeholder
        return init_states

    def update(self, encodings: torch.Tensor, actions: torch.Tensor):
        new_encodings = encodings.clone()
        n, m = encodings.shape
        update_success = torch.zeros(n, dtype=torch.bool)
        ind_mask = (new_encodings == self.placeholder).long()
        indices = ind_mask.argmax(axis=1)
        new_encodings[torch.arange(len(encodings)), indices] = actions
        # setting new children locations to be placeholder
        new_action_arities = self.action_arities[actions]
        left_idx = indices * 2 + 1
        right_idx = left_idx + 1
        # updates of constants/features is always successful
        update_success[new_action_arities == 0] = True

        is_unary = new_action_arities == 1
        is_unary[is_unary.clone()] = (left_idx[is_unary] < m)
        unary_success = (new_encodings[is_unary, left_idx[is_unary]] == -1)
        is_unary[is_unary.clone()] = unary_success
        update_success[is_unary] = True
        new_encodings[is_unary, left_idx[is_unary]] = self.placeholder

        is_binary = new_action_arities == 2
        is_binary[is_binary.clone()] = (right_idx[is_binary] < m)
        binary_success = (new_encodings[is_binary, left_idx[is_binary]] == -1) & \
            (new_encodings[is_binary, right_idx[is_binary]] == -1)
        is_binary[is_binary.clone()] = binary_success
        update_success[is_binary] = True
        new_encodings[is_binary, left_idx[is_binary]] = self.placeholder
        new_encodings[is_binary, right_idx[is_binary]] = self.placeholder

        new_encodings[update_success, indices[update_success]] = actions[update_success]
        return new_encodings, update_success

    def mask(self, encodings: torch.Tensor):
        n = len(encodings)
        ecd_mask = (encodings == self.placeholder)
        indices = ecd_mask.long().argmax(axis=1)
        # identify complete trees
        done_idx = ~ecd_mask.any(axis=1)

        mask = torch.zeros((n, self.num_actions))
        mask[done_idx, -1] = 1
        mask[~done_idx, :-1] = 1

        left_idx = indices * 2 + 1
        # RULE 1: check whether the most recent token is leaf or not
        is_leaf = left_idx >= encodings.shape[1]
        mask[is_leaf, self.action_space.feat_num:-1] = 0
        # RULE 2: check whether the parent is unary, in which case we disallow constant
        par_idx = torch.div(indices - 1, 2, rounding_mode='floor').clamp(min=0)
        min_action_idx = self.action_space.feat_num + self.action_space.op_num
        is_par_unary = encodings[torch.arange(n), par_idx] < min_action_idx
        mask[is_par_unary, 0] = 0
        # RULE 3: left sibling cannot be constant already
        has_left_sib = (indices % 2 == 0) & (indices > 0)
        sub_mask = mask[has_left_sib, :]
        left_constant_idx = encodings[has_left_sib, indices[has_left_sib] - 1] == 0
        sub_mask[left_constant_idx, 0] = 0
        mask[has_left_sib, :] = sub_mask
        return mask, done_idx

    def calc_loss(self, encodings: torch.Tensor):
        """
        ! we assume that `encodings` here are valid (complete) trees
        """
        n = len(encodings)
        loss = torch.zeros(n)
        cache_miss = []
        expressions = []

        for i in range(n):
            cache_loss = self.loss_buffer.get(encodings[i])
            if cache_loss is None:
                cache_miss.append(i)
                expression = ExpressionTree(encodings[i], self.action_fns, self.action_arities, self.action_names)
                expressions.append(expression)
            else:
                loss[i] = cache_loss

        # perform inner optimization
        ensemble = ETEnsemble(expressions)
        optimize_constant(ensemble, self.X, self.y, self.inner_loop_config)

        # compute final loss for expressions not cached
        with torch.no_grad():
            for i, miss_idx in enumerate(cache_miss):
                final_reward = self.criterion(expressions[i](self.X), self.y)
                if not torch.isfinite(final_reward):
                    loss[miss_idx] = torch.inf
                else:
                    loss[miss_idx] = final_reward
                has_constant = len(expressions[i].constants) > 0
                self.loss_buffer.set(encodings[miss_idx], has_constant, loss[miss_idx])
        return loss

    def reward(self, encodings: torch.Tensor, is_eval: bool = False):
        if not self.reward_manager:
            self.set_up_reward_manager()

        loss = self.calc_loss(encodings)
        rewards = self.reward_manager.calc_rewards(loss, encodings, is_eval)
        return rewards
