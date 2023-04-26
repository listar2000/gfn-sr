import torch
from policy import RNNForwardPolicy, CanonicalBackwardPolicy
from actions import Action

test_action = Action(num_features=2)


def test_forward_policy():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    fp = RNNForwardPolicy(3, 32, 5, device=device).to(device)
    mock_input = torch.tensor([[2, 3, -2], [-2, -1, -1], [-1, -2, -2]])
    probs = fp(mock_input)
    assert ((probs.sum(axis=1) - 1.0) <= 1e-5).all()


def test_backward_policy():
    bp = CanonicalBackwardPolicy(5)
    mock_input = torch.tensor([[10, 2, -1], [10, 0, 1], [0, -1, -1]])
    probs = bp.forward(mock_input)
    assert (probs == torch.tensor([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])).all()

