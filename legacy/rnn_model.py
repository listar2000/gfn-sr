import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sib_and_par(index: int):
    # Get the parent index
    parent_index = (index - 1) // 2

    # Calculate the sibling index
    if index % 2 == 0:
        sibling_index = index - 1
    else:
        sibling_index = 0

    return sibling_index, parent_index


class FixedTreeRNN(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, num_layers=1, template=None):
        super(FixedTreeRNN, self).__init__()

        self.hidden = None
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.template = template

        self.rnn = nn.RNN(state_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, s):
        # print(s.shape)
        s0 = s[0, :] > 0
        idx = (self.template ^ s0).argmax()
        sib_par = s[:, get_sib_and_par(idx)].unsqueeze(1)
        # print(sib_par.shape)
        if self.hidden is not None:
            self.hidden = self.init_hidden(len(s))
        # Key info should be shaped (batch_size, seq_length, input_size)
        output, self.hidden = self.rnn(sib_par, self.hidden)
        output = self.fc(output[:, -1, :])  # Get the last output in the sequence
        probabilities = F.softmax(output, dim=1)

        return probabilities

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# Example usage
input_size = 10
hidden_size = 128
output_size = 5
num_layers = 1

model = FixedTreeRNN(input_size, hidden_size, output_size, num_layers)
