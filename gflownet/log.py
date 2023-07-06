import torch


class Log:
    def __init__(self, traj, fwd_probs, rewards, total_flow):
        """
        Initializes a Stats object to record sampling statistics from a
        GFlowNet (trajectories, forward probabilities, and rewards)
        
        Args:
            traj: The trajectory of state evolution
            
            fwd_probs: The forward probabilities for sampling actions in state evolution
            
            rewards: The rewards for the complete samples
        """
        self._traj = traj
        self._fwd_probs = fwd_probs
        self.rewards = rewards
        self.total_flow = total_flow
        self._actions = []
    
    @property
    def traj(self):
        if type(self._traj) is list:
            self._traj = torch.cat(self._traj, dim=1)[:, :-1, :]
        return self._traj
    
    @property
    def fwd_probs(self):
        if type(self._fwd_probs) is list:
            self._fwd_probs = torch.cat(self._fwd_probs, dim=1)
        return self._fwd_probs
    
    @property
    def actions(self):
        raise NotImplementedError("this method is not supported now")
        # if type(self._actions) is list:
        #     self._actions = torch.cat(self._actions, dim=1)
        # return self._actions
    
    @property
    def back_probs(self):
        raise NotImplementedError("this method is not supported now")
        # if self._back_probs is not None:
        #     return self._back_probs
        #
        # s = self.traj[:, 1:, :].reshape(-1, self.env.state_dim)
        # prev_s = self.traj[:, :-1, :].reshape(-1, self.env.state_dim)
        # actions = self.actions[:, :-1].flatten()
        #
        # terminated = (actions == -1) | (actions == self.env.num_actions - 1)
        # zero_to_n = torch.arange(len(terminated))
        # back_probs = self.backward_policy(s) * self.env.mask(prev_s)[0]
        # back_probs = torch.where(terminated, 1, back_probs[zero_to_n, actions])
        # self._back_probs = back_probs.reshape(self.num_samples, -1)
        #
        # return self._back_probs
