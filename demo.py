import torch

def model2(x, a, b):
    return (a + b) * (torch.log(a) + torch.exp(b * x))

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

torch.manual_seed(0)

x = torch.linspace(-1, 1, 100)
y_true = model2(x, torch.tensor([2]), torch.tensor([1]))

a = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

def lbfgs_step():
    optimizer.zero_grad()
    y_pred = model2(x, a, b)
    loss = mse_loss(y_pred, y_true)
    loss.backward()
    return loss

optimizer = torch.optim.Adam([a, b], lr=1e-3)

for i in range(1000):
    optimizer.step(lbfgs_step)
    print(f"Epoch {i + 1}: {a}, {b}")

import matplotlib.pyplot as plt

plt.plot(x, y_true, color="red")
y_pred = model2(x, a, b).detach().numpy()
plt.plot(x, y_pred, color="blue")
plt.show()