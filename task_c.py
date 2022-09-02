import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).reshape(-1, 2)
y_train = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)

class SigmoidRegressionModel:

    def __init__(self):
        self.W_1 = torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
        self.b_1 = torch.tensor([[-5.0, 15]], requires_grad=True)
        self.W_2 = torch.tensor([[10.0], [10.0]], requires_grad=True)
        self.b_2 = torch.tensor([[-18.0]], requires_grad=True)

    def f1(self, x):
        return torch.sigmoid(x @ self.W_1 + self.b_1)

    def f2(self, x):
        return torch.sigmoid(x @ self.W_2 + self.b_2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = SigmoidRegressionModel()

optimizer = torch.optim.SGD([model.W_1, model.b_1, model.W_2, model.b_2], 0.05)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W_1 = %s, b_1 = %s, W_2 = %s, b_2 = %s, loss = %s" % (model.W_1.data, model.b_1.data, model.W_2.data, model.b_2.data, model.loss(x_train, y_train).data))

# Visualize result
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(), 'o', label='$(x^{(i)},y^{(i)})$')

grid_points = 10
x_grid, z_grid = np.meshgrid(np.linspace(-0.1, 1.1, grid_points), np.linspace(-0.1, 1.1, grid_points))
y_grid = np.empty([grid_points, grid_points])

for i in range(0, x_grid.shape[0]):
    for j in range(0, x_grid.shape[1]):
        tens = torch.tensor([np.float32(x_grid[i, j]), np.float(z_grid[i, j])])
        tens.double()
        y_grid[i, j] = model.f(tens)

ax.plot_wireframe(x_grid, z_grid, y_grid, color='green')
plt.show()