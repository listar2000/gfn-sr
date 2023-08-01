import torch

from actions import ExpressionTree, optimize_constant
from env import SRTree


class RewardManager(object):
    """
    A reward manager is in charge of calculating the reward from certain reward functions
    """
    def __init__(self, env: SRTree, verbose: bool = True):
        self.best_reward = -torch.inf
        self.best_expr = None

        self.env = env
        self.verbose = verbose

    def _update_rewards(self, rewards: torch.Tensor, loss: torch.Tensor, encodings: torch.Tensor):
        if len(rewards) > 1 and torch.max(rewards) > self.best_reward:
            best_reward_vanilla = torch.max(rewards)
            best_action = torch.argmax(rewards)
            best_expr = ExpressionTree(encodings[best_action],
                                       self.env.action_fns, self.env.action_arities, self.env.action_names)
            optimize_constant(best_expr, self.env.X, self.env.y, self.env.inner_loop_config)
            loss_opt = self.env.criterion(best_expr(self.env.X), self.env.y)
            if self.verbose:
                print(f"\nnew best reward (vanilla): {best_reward_vanilla}")
                print(f"mse (pre/post optimized): {loss[best_action]}/{loss_opt}")
                print(f"expr: {str(best_expr)}")
            self.best_reward = best_reward_vanilla
            self.best_expr = best_expr

    def calc_rewards(self, loss: torch.Tensor, encodings: torch.Tensor):
        raise NotImplementedError("Cannot use an abstract reward manager class")


class NRMSEReward(RewardManager):
    def __init__(self, env: SRTree, verbose: bool = True):
        super(NRMSEReward, self).__init__(env, verbose)

    def calc_rewards(self, loss: torch.Tensor, encodings: torch.Tensor, is_eval: bool = False):
        nrmse = torch.sqrt(loss) / torch.std(self.env.y)
        rewards = torch.clamp(self.env.loss_thres / (self.env.loss_thres + nrmse), min=0.01)
        if not is_eval:
            self._update_rewards(rewards, loss, encodings)
        return rewards


class TSSReward(RewardManager):
    def __init__(self, env: SRTree, verbose: bool = True):
        super(TSSReward, self).__init__(env, verbose)
        self.max_mse = ((self.env.y - self.env.y.mean()) ** 2).mean()

    def calc_rewards(self, loss: torch.Tensor, encodings: torch.Tensor, is_eval: bool = False):
        rewards = torch.clamp(1.0 - loss / self.max_mse, min=0.01)
        if not is_eval:
            self._update_rewards(rewards, loss, encodings)
        return rewards


class DynamicTSSReward(RewardManager):
    def __init__(self, env: SRTree, verbose: bool = True):
        super(DynamicTSSReward, self).__init__(env, verbose)
        self.baseline_mse = ((self.env.y - self.env.y.mean()) ** 2).mean()
        self.gamma = 0.95
        self.q = 0.1

    def calc_rewards(self, loss: torch.Tensor, encodings: torch.Tensor, is_eval: bool = False):
        rewards = torch.clamp(1.0 - loss / self.baseline_mse, min=1e-4)
        if not is_eval:
            self._update_rewards(rewards, loss, encodings)
        new_baseline_mse = self.gamma * self.baseline_mse + (1 - self.gamma) * torch.quantile(loss, q=self.q)
        if new_baseline_mse < self.baseline_mse:
            self.baseline_mse = new_baseline_mse
        return rewards


class StructureReward(RewardManager):
    def __init__(self, env: SRTree, verbose: bool = True):
        super(StructureReward, self).__init__(env, verbose)

    def calc_rewards(self, loss: torch.Tensor, encodings: torch.Tensor, is_eval: bool = False):
        num_elements = (encodings > -1).sum(axis=1)
        rewards = torch.ones(len(encodings))
        rewards[num_elements > 1] = 1e-4
        if not is_eval:
            self._update_rewards(rewards, loss, encodings)
        return rewards
