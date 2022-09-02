import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([1.0, 0.0]).reshape(-1, 1)
y_train = torch.tensor([0.0, 1.0]).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.5)
for epoch in range(500):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()