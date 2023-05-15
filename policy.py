import torch
from torch import nn
import torch.nn.functional as F
from actions import get_next_node_indices


class RNNForwardPolicy(nn.Module):
    def __init__(self, batch_size, hidden_dim, num_actions,
                 num_layers=1, model='rnn', dropout=0.0, placeholder=-2, one_hot=True, device=None):
        super(RNNForwardPolicy, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.dropout = dropout
        self.placeholder = placeholder
        self.one_hot = one_hot
        self.device = torch.device("cpu") if not device else device

        # if using one_hot, we turn (sibling, parent) to 2 * num_actions vector
        state_dim = 2 * num_actions if self.one_hot else 2

        if model == 'rnn':
            self.rnn = nn.RNN(state_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout).to(device)
        elif model == 'gru':
            self.rnn = nn.GRU(state_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout).to(device)
        else:
            raise NotImplementedError("unsupported model: " + model)

        self.fc = nn.Linear(hidden_dim, num_actions)
        # self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, self.hidden_size), requires_grad=True) \
        #     .to(self.device)
        self.init_hidden = nn.Parameter(data=torch.zeros(self.num_layers, self.hidden_size), requires_grad=True) \
            .to(self.device)

    def actions_to_one_hot(self, siblings, parents):
        sibling_oh = F.one_hot(siblings, num_classes=self.num_actions)
        parent_oh = F.one_hot(parents, num_classes=self.num_actions)
        return torch.cat((sibling_oh, parent_oh), axis=1)

    def forward(self, encodings):
        if encodings[0, 0] == self.placeholder:
            self.hidden = self.init_hidden.repeat(1, len(encodings), 1)

        nodes_to_assign, siblings, parents = get_next_node_indices(encodings, self.placeholder)
        if self.one_hot:
            rnn_input = self.actions_to_one_hot(siblings, parents)
        else:
            rnn_input = torch.stack([siblings, parents], axis=1)

        # match dimension of the hidden state
        rnn_input = rnn_input.unsqueeze(1).float()
        output, self.hidden = self.rnn(rnn_input, self.hidden)
        # Get the last output in the sequence
        output = self.fc(output[:, -1, :])
        probabilities = F.softmax(output, dim=1)

        return probabilities


class CanonicalBackwardPolicy(nn.Module):
    def __init__(self, num_actions: int):
        super(CanonicalBackwardPolicy, self).__init__()
        self.num_actions = num_actions

    def forward(self, encodings: torch.Tensor):
        """
        Calculate the backward probability matrix for a given encoding.
        This downgrades into simply finding the recent action assigned in
        the forward pass due to the tree structure of our environment.
        Let (M, T, A) be the (batch size, max tree size, action space dim)
        Args:
            encodings: a (M * T) encoding matrix
        Returns:
            probs: a (M * A) probability matrix
        """
        ecd_mask = (encodings >= 0)
        assert (ecd_mask.sum(axis=1) >= 0).all()
        # get the indices of the recently assigned node using a special trick
        # we want to get the last `True` element of each row, so we multiply the bool
        # with a value that increases with column index, then taking the argmax
        indices = (ecd_mask * torch.arange(1, encodings.shape[1] + 1)).argmax(axis=1)
        actions = encodings[torch.arange(len(encodings)), indices]
        probs = F.one_hot(actions, self.num_actions)
        return probs