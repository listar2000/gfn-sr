import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from .log import Log


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        """
        Initializes a GFlowNet using the specified forward and backward policies
        acting over an environment, i.e. a state space and a reward function.
        
        Args:
            forward_policy: A policy network taking as input a state and
            outputting a vector of probabilities over actions
            
            backward_policy: A policy network (or fixed function) taking as
            input a state and outputting a vector of probabilities over the
            actions which led to that state
            
            env: An environment defining a state space and an associated reward
            function
        """
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
        # the total_flow is a learnable param
        self.total_flow = Parameter(torch.ones(1))
    
    def mask_and_normalize(self, s, probs):
        """
        Masks a vector of action probabilities to avoid illegal actions (i.e.
        actions that lead outside the state space).
        
        Args:
            s: An NxD matrix representing N states
            
            probs: An NxA matrix of action probabilities
        """
        # 1e-8 for smoothing and avoiding division by zero
        mask, done_idx = self.env.mask(s)
        probs = mask * (probs + 1e-8)
        probs = probs / probs.sum(1).unsqueeze(1)
        return probs, done_idx
    
    def forward_probs(self, s):
        """
        Returns a vector of probabilities over actions in a given state.
        
        Args:
            s: An NxD matrix representing N states
        """
        probs = self.forward_policy(s)
        return self.mask_and_normalize(s, probs)
    
    def sample_states(self, s0):
        """
        Samples and returns a collection of final states from the GFlowNet.
        
        Args:
            s0: An NxD matrix of initial states
        """
        s, n = s0.clone(), len(s0)
        done = torch.zeros(n, dtype=torch.bool)

        _traj, _fwd_probs = [s0.view(n, 1, -1)], []
        while not done.all():
            probs, done = self.forward_probs(s)
            probs = probs[~done]
            actions = Categorical(probs).sample()
            state, update_success = self.env.update(s[~done], actions)
            all_success = update_success.all()

            fwd_probs = torch.ones(n, 1)
            if all_success:
                s[~done] = state
                fwd_probs[~done] = probs.gather(1, actions.unsqueeze(1))
            else:
                s[~done][update_success] = state[update_success]
                fwd_probs[~done][update_success] = probs.gather(1, actions.unsqueeze(1))[update_success]

            # logging necessary information
            _traj.append(s.clone().view(n, 1, -1))
            _fwd_probs.append(fwd_probs)

        _rewards = self.env.reward(s)
        log = Log(_traj, _fwd_probs, _rewards, self.total_flow)
        return s, log
    
    def evaluate_trajectories(self, traj, actions):
        """
        Returns the GFlowNet's estimated forward probabilities, backward
        probabilities, and rewards for a collection of trajectories. This is
        useful in an offline learning context where samples drawn according to
        another policy (e.g. a random one) are used to train the model.
        
        Args:
            traj: The trajectory of each sample
            
            actions: The actions that produced the trajectories in traj
        """
        num_samples = len(traj)
        traj = traj.reshape(-1, traj.shape[-1])
        actions = actions.flatten()
        finals = traj[actions == self.env.num_actions - 1]
        zero_to_n = torch.arange(len(actions))
        
        fwd_probs = self.forward_probs(traj)
        fwd_probs = torch.where(actions == -1, 1, fwd_probs[zero_to_n, actions])
        fwd_probs = fwd_probs.reshape(num_samples, -1)
        
        actions = actions.reshape(num_samples, -1)[:, :-1].flatten()
        
        back_probs = self.backward_policy(traj)
        back_probs = back_probs.reshape(num_samples, -1, back_probs.shape[1])
        back_probs = back_probs[:, 1:, :].reshape(-1, back_probs.shape[2])
        back_probs = torch.where((actions == -1) | (actions == 2), 1,
                                 back_probs[zero_to_n[:-num_samples], actions])
        back_probs = back_probs.reshape(num_samples, -1)
        
        rewards = self.env.reward(finals)
        
        return fwd_probs, back_probs, rewards
