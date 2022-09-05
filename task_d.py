import torch
import torchvision
import matplotlib.pyplot as plt

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)

x_train = mnist_train.data.reshape(-1, 784).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

x_test = mnist_test.data.reshape(-1, 784).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

class SoftmaxModel:

    def __init__(self):
        self.W = torch.zeros(784, 10, requires_grad=True)
        self.b = torch.tensor([0.0], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return torch.nn.functional.\
            binary_cross_entropy_with_logits(self.logits(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = SoftmaxModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.000125)
for epoc in range(150):
    model.loss(x_train,  y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))

plt_grid_rows = 5
plt_grid_columns = 2

f, plt_grid = plt.subplots(plt_grid_rows, plt_grid_columns)
# Show the input of the first observation in the training set
for i in range(0, plt_grid_rows):
    for j in range(0, plt_grid_columns):
        index = (i * plt_grid_columns) + j
        image = model.W.detach()[:, index].reshape(28, 28)
        plt_grid[i, j].imshow(image)
        plt.imsave(f'x_train_{index}.png', image)

# Save the input of the first observation in the training set
plt.imsave('x_train_1.png', x_train[0, :].reshape(28, 28))

acc = model.accuracy(x_test, y_test)
print("Accuracy: %s" % acc)
plt.show()
