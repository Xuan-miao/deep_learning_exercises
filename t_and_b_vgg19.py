import sys

import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils import data
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

EPOCHS = 20


class Metrics:
    def __init__(self, n):
        self.metrics = [0.0] * n

    def mark(self, *args):
        self.metrics = [a + float(b) for a, b in zip(self.metrics, args)]

    def __getitem__(self, item):
        return self.metrics[item]


def draw(x, y, label):
    plt.plot(range(1, x + 1), y[0], label=label[0])
    plt.plot(range(1, x + 1), y[1], label=label[1])
    plt.plot(range(1, x + 1), y[2], label=label[2])
    plt.legend()
    plt.xlim(1, x)
    plt.show()


def eval_net(model, data_loader, epoch, device='cpu'):
    model.eval()
    metrics = Metrics(2)
    for xx, yy in data_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        with torch.no_grad():
            yy_hat = net(xx).max(1).indices
        r = sum(yy == yy_hat)
        pred_pos = sum(yy_hat == 1)
        tp = sum(yy_hat * yy)
        fp = pred_pos - tp
        fn = 30 - tp
        P = tp / pred_pos
        R = tp / (tp + fn)
        f1 = 2 * P * R / (P + R)
        if epoch == EPOCHS - 1:
            # print(f'yy---- {yy}\nyy_hat {yy_hat}\nyy*hat {yy * yy_hat}')
            print(f'precision {P:f}, recall {R:f}, f1 {f1:f}')
        metrics.mark(r, len(yy))
    return metrics[0] / metrics[1]


def train_net(model, train_iter, test_iter, only_fc=True,
              optimizer_cls=torch.optim.Adam, loss_fn=nn.CrossEntropyLoss(),
              epochs=10, device='cpu'):
    train_loss = []
    train_acc = []
    test_acc = []
    if only_fc:
        optimizer = optimizer_cls(model.classifier.parameters())
    else:
        optimizer = optimizer_cls(model.parameters())
    for epoch in range(epochs):
        metrics = Metrics(3)
        model.train()
        for i, (xx, yy) in enumerate(train_iter):
            xx = xx.to(device)
            yy = yy.to(device)
            yy_hat = model(xx)
            loss = loss_fn(yy_hat, yy)
            # loss = nn.CrossEntropyLoss()(yy_hat, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            r = sum(yy == (yy_hat.max(1).indices))
            metrics.mark(loss.item(), r, len(yy))
        train_loss.append(metrics[0] / i)
        train_acc.append(metrics[1] / metrics[2])
        test_acc.append(eval_net(model, test_iter, epoch, device))
        print(f'Epoch {epoch + 1}, train_loss {train_loss[-1]:f} ,'
              f'train_acc {train_acc[-1]:f}, test_acc {test_acc[-1]:f}')
    draw(epochs, [train_loss, train_acc, test_acc],
         ['train_loss', 'train_acc', 'test_acc'])


train_aug = transforms.Compose(
    [transforms.RandomCrop(224), transforms.ToTensor(),
     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),

     transforms.RandomHorizontalFlip()])
test_aug = transforms.Compose(
    [transforms.RandomCrop(224), transforms.ToTensor()])
train_img = ImageFolder('C:/data/taco_and_burrito/train', transform=train_aug)
test_img = ImageFolder('C:/data/taco_and_burrito/test', transform=test_aug)
train_loader = data.DataLoader(train_img, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_img, batch_size=60, shuffle=True)
# print(train_img, '\n-----------\n', train_img.classes, train_img.class_to_idx)

net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
for p in net.features.parameters():
    p.requires_grad = False
net.classifier[6] = nn.Linear(4096, 2)
print(net)
# sys.exit()

net.to('cuda:0')
train_net(net, train_loader, test_loader, epochs=EPOCHS, device='cuda:0')
# torch.save(net.state_dict(), 'vgg16_weights.pth')
