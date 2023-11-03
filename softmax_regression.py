import sys
import torch
import matplotlib.pyplot as plt

from torch.utils import data
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms

batch_size = 256
trans = transforms.ToTensor()
mnist_train = FashionMNIST(
    root='../data', train=True, transform=trans, download=True)
mnist_test = FashionMNIST(
    root='../data', train=False, transform=trans, download=True)
train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

conv_net = nn.Sequential(nn.Conv2d(1, 32, 5),
                         nn.MaxPool2d(2),
                         nn.ReLU(),
                         nn.BatchNorm2d(32),
                         nn.Dropout2d(0.25),
                         nn.Conv2d(32, 64, 5),
                         nn.MaxPool2d(2),
                         nn.ReLU(),
                         nn.BatchNorm2d(64),
                         nn.Dropout2d(0.25),
                         nn.Flatten())
test_input = torch.ones(1, 1, 28, 28)
conv_output_size = conv_net(test_input).size()[-1]
mlp = nn.Sequential(nn.Linear(conv_output_size, 200),
                    nn.ReLU(),
                    nn.BatchNorm1d(200),
                    nn.Dropout(0.25),
                    nn.Linear(200, 10))
# net = nn.Sequential(conv_net, mlp)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

train_losses = []
running_losses = []
train_acc = []
val_acc = []
optimizer = torch.optim.Adam(net.parameters())
net.to('cuda:0')


def eval_net(net, data_loader, device='cpu'):
    net.eval()
    y_list = []
    y_pred_list = []
    for x, y in test_loader:
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        y_list.append(y)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        y_pred_list.append(y_pred)
    y_list = torch.cat(y_list)
    y_pred_list = torch.cat(y_pred_list)
    acc = (y_list == y_pred_list).float().sum() / len(y_list)
    return acc.item()


# sys.exit()

for epoch in range(10):
    running_loss = 0.0
    net.train()
    n = 0
    n_acc = 0
    for i, (xx, yy) in enumerate(train_loader):
        xx = xx.to('cuda:0')
        # print(xx.shape[0])
        yy = yy.to('cuda:0')
        h = net(xx)
        # print(yy.shape)
        loss = nn.CrossEntropyLoss()(h, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n += xx.shape[0]
        _, y_pred = h.max(1)
        n_acc += (yy == y_pred).float().sum().item()
    train_acc.append(n_acc / n)
    train_losses.append(loss.item())
    running_losses.append(running_loss / i)
    val_acc.append(eval_net(net, test_loader, device='cuda:0'))
    print(f'Epoch {epoch + 1}, loss {running_losses[-1]:f}, ', end='')
    print(f'train_acc {train_acc[-1]:f}, test_acc {val_acc[-1]:f}')

plt.plot(range(10), train_acc, label='train_acc')
plt.plot(range(10), val_acc, label='test_acc')
plt.plot(range(10), running_losses, label='running')
plt.legend()
plt.show()
