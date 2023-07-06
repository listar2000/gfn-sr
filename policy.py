import torch
from torch import nn
import torch.nn.functional as F
from actions import get_next_node_indices

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEquationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions, num_layers=4, num_heads=4, dropout=0.1):
        super(TransformerEquationPredictor, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=hidden_size*4, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        print(x)
        embedded = self.embedding(x)
        print(embedded)
        embedded = self.positional_encoding(embedded)
        padding_token = 0 # or -1???
        hidden = self.transformer(embedded, embedded, src_mask=(x != padding_token).unsqueeze(-2))
        output = self.fc(hidden)

        output = F.softmax(output, dim=2)  # Apply softmax to the output predictions
        # predicted_token = torch.argmax(output, dim=2)[:, -1]

        return output

class TransformerForwardPolicyVanilla(nn.Module):
    def __init__(self, batch_size, hidden_size, feed_forward_size, num_actions, model='Transformer',
                 num_layers=4, num_heads=4, placeholder=-2, one_hot=True, device=None):
        super(TransformerForwardPolicyVanilla, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.placeholder = placeholder
        self.one_hot = one_hot
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # if using one_hot, we turn (sibling, parent) to 2 * num_actions + 2 vector
        # where the additional 2 denotes 2 placeholder symbols
        state_dim = 2 * num_actions + 2 if self.one_hot else 2

        self.embedding = nn.Embedding(state_dim, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, feed_forward_size),
            num_layers
        )
        self.decoder = nn.Linear(hidden_size, state_dim)

        self.fc = nn.Linear(state_dim, num_actions).to(self.device)

    def actions_to_one_hot(self, siblings, parents):
        # leave the first
        siblings[siblings == self.placeholder] = -1
        parents[parents == self.placeholder] = -1
        sibling_oh = F.one_hot(siblings + 1, num_classes=self.num_actions + 1)
        parent_oh = F.one_hot(parents + 1, num_classes=self.num_actions + 1)
        # print("shape", torch.cat((sibling_oh, parent_oh), axis=1).shape)
        return torch.cat((sibling_oh, parent_oh), axis=1)

    def forward(self, encodings):
        nodes_to_assign, siblings, parents = get_next_node_indices(encodings, self.placeholder)
        if self.one_hot:
            input_seq = self.actions_to_one_hot(siblings, parents).to(self.device)
        else:
            input_seq = torch.stack([siblings, parents], axis=1).to(self.device)

        embedded = self.embedding(input_seq)
        encoded = self.encoder(embedded)
        output = self.decoder(encoded)

        output = self.fc(output[:, -1, :])
        probabilities = F.softmax(output, dim=-1)
        # print("input seq shape", input_seq.shape)
        # print("probabilities shape", probabilities.shape)
        return probabilities.cpu()


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
        self.model = model

        # if using one_hot, we turn (sibling, parent) to 2 * num_actions + 2 vector
        # where the additional 2 denotes 2 placeholder symbols
        state_dim = 2 * num_actions + 2 if self.one_hot else 2

        if model == 'rnn':
            self.rnn = nn.RNN(state_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout).to(self.device)
        elif model == 'gru':
            self.rnn = nn.GRU(state_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=self.dropout).to(self.device)
        elif model == 'lstm':
            self.rnn = nn.LSTM(state_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=self.dropout).to(self.device)
            self.init_c0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
        else:
            raise NotImplementedError("unsupported model: " + model)

        self.fc = nn.Linear(hidden_dim, num_actions).to(self.device)
        self.init_h0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)

    def actions_to_one_hot(self, siblings, parents):
        # leave the first
        siblings[siblings == self.placeholder] = -1
        parents[parents == self.placeholder] = -1
        sibling_oh = F.one_hot(siblings + 1, num_classes=self.num_actions + 1)
        parent_oh = F.one_hot(parents + 1, num_classes=self.num_actions + 1)
        print("shape", torch.cat((sibling_oh, parent_oh), axis=1).shape)
        return torch.cat((sibling_oh, parent_oh), axis=1)

    def forward(self, encodings):
        if encodings[0, 0] == self.placeholder:
            self.h0 = self.init_h0.unsqueeze(1).repeat(1, len(encodings), 1)
            if self.model == 'lstm':
                self.c0 = self.init_c0.unsqueeze(1).repeat(1, len(encodings), 1)

        nodes_to_assign, siblings, parents = get_next_node_indices(encodings, self.placeholder)
        if self.one_hot:
            rnn_input = self.actions_to_one_hot(siblings, parents).to(self.device)
        else:
            rnn_input = torch.stack([siblings, parents], axis=1).to(self.device)

        # match dimension of the hidden state
        rnn_input = rnn_input.unsqueeze(1).float()

        rnn_input = rnn_input.float()
        if self.model == 'lstm':
            output, (self.h0, self.c0) = self.rnn(rnn_input, (self.h0, self.c0))
        else:
            output, self.h0 = self.rnn(rnn_input, self.h0)
        # Get the last output in the sequence
        output = self.fc(output[:, -1, :])
        probabilities = F.softmax(output, dim=1)

        print("input seq shape", rnn_input.shape)
        print("probabilities shape", probabilities.shape)
        return probabilities.cpu()


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
