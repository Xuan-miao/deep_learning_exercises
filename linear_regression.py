import time
import numpy as np
import torch

from torch import nn
from torch.utils import data


class Timer:
    def __init__(self):
        self.times = []
        self.tick = time.time()  # start timer

    def stop(self):
        self.times.append(time.time() - self.tick)
        return self.times[-1]

    def avg_time(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def acc_sum(self):
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_example):
    x = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape(-1, 1)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, batch_size)

net = nn.Sequential(nn.Linear(2, 1))
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epoch = 3
for epoch in range(num_epoch):
    for X, y in data_iter:
        loss = nn.MSELoss()(net(X), y)
        trainer.zero_grad()
        loss.backward()
        trainer.step()
    loss = nn.MSELoss()(net(features), labels)
    print('epoch ', epoch, ', loss ', float(loss))
    print(f'epoch {epoch + 1}, loss {loss:f}')
