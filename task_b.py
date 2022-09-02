import torch
import matplotlib.pyplot as plt
import numpy as np

x_train = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).reshape(-1, 2)
y_train = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(5000):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))

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