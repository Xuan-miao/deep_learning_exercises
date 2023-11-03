import sys
import torch
import matplotlib.pyplot as plt
import torchvision.datasets

from torch import nn
from torch.utils import data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights


def eval_net(model, data_loader, device='cpu'):
    model.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()


def train_net(net, train_loader, test_loader, only_fc=True,
              optimizer_cls=torch.optim.Adam, loss_fn=nn.CrossEntropyLoss(),
              epochs=10, device='cpu'):
    train_losses = []
    train_acc = []
    test_acc = []
    if only_fc:
        optimizer = optimizer_cls(net.fc.parameters())
    else:
        optimizer = optimizer_cls(net.parameters())
    for epoch in range(epochs):
        running_loss = 0.0
        n = 0
        n_acc = 0
        net.train()
        for i, (xx, yy) in enumerate(train_loader):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            # loss = nn.CrossEntropyLoss()(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        train_acc.append(n_acc / n)
        test_acc.append(eval_net(net, test_loader, device))
        print(f'Epoch {epoch + 1}, loss {train_losses[-1]:f}, '
              f'train_acc {train_acc[-1]:f}, test_acc {test_acc[-1]:f}')
    x = [i + 1 for i in range(epochs)]
    plt.plot(x, train_losses, label='train_loss')
    plt.plot(x, train_acc, label='train_acc')
    plt.plot(x, test_acc, label='test_acc')
    plt.xlim((1, epochs))
    plt.legend()
    plt.show()


trans = transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()])
train_img = ImageFolder('../data/taco_and_burrito/train', transform=trans)
test_img = ImageFolder('../data/taco_and_burrito/test', transform=trans)
train_loader = data.DataLoader(train_img, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_img, batch_size=32, shuffle=True)
# print(train_img, '\n-----------\n', train_img.classes, train_img.class_to_idx)

# net = models.resnet18(pretrained=True)
net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# print(net)
# sys.exit()
for p in net.parameters():
    p.requires_grad = False
fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim, 2)

x, y = next(iter(test_loader))
print(x.shape, y.shape)
print(y[:10])
# plt.imshow(x[1].transpose(0, 1).transpose(1, 2))
plt.imshow(x[1].permute(1, 2, 0))
# plt.imshow(torch.einsum('ijk->jki', x[1]))
plt.show()
# sys.exit()

net.to('cuda:0')
train_net(net, train_loader, test_loader, epochs=20, device='cuda:0')
