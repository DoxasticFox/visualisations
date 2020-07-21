#!/usr/bin/env python3

import itertools
import matplotlib.pyplot as plt
import torch

data = torch.randn(1000, 2)
#data = (torch.randn(1000, 2) + 0.5).sin()

# x = data[:, 0].round()
# y = data[:, 1]

# div = (data[:, 0].square() + data[:, 1].square()).sqrt()
# x = data[:, 0] / div
# y = data[:, 1] / div

# x = data[:, 0]
# y = (data[:, 0] * 3.0).sin()

x, y = [], []
for x_, y_ in data:

    if x_ < 0.0:
        x_ = x_ + 1.0
        div = (x_ ** 2.0 + y_ ** 2.0) ** 0.5
        x_ = x_ / div * 0.50 - 0.75
        y_ = y_ / div * 1.25 + 0.50
    else:
        x_ = x_ - 1.0
        div = (x_ ** 2.0 + y_ ** 2.0) ** 0.5
        x_ = x_ / div * 0.75 + 0.75
        y_ = y_ / div * 0.75 - 0.75

    x.append(x_)
    y.append(y_)
x, y = torch.tensor(x), torch.tensor(y)


class ResSin(torch.nn.Module):
    def __init__(self, out_features, bendy_bit_lr=0.5):
        super(ResSin, self).__init__()

        self.bendiness = torch.nn.Parameter(torch.zeros(out_features))

        self.fix_bias = torch.nn.Parameter(
            torch.randn(out_features),
            requires_grad=False,
        )
        self.var_bias = torch.nn.Parameter(
            torch.zeros(out_features),
        )

        pi = (torch.acos(torch.tensor(0.0)) * 2.0).item()
        self.fix_weight = torch.nn.Parameter(
            torch.ones(out_features) * pi
        )
        self.var_weight = torch.nn.Parameter(
            torch.randn(out_features) * pi * 2
        )

        self.out_features = out_features
        self.bendy_bit_lr = bendy_bit_lr

    def forward(self, x):
        weight = \
            self.fix_weight + \
            self.var_weight
        bias = \
            self.fix_bias + \
            self.var_bias
        return \
            self.bendiness * self.bendy_bit_lr * (x * weight + bias).sin() + \
            x


class ResLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ResLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            ResLinear._initial_weight(in_features, out_features)
        )

    @staticmethod
    @torch.no_grad()
    def _initial_weight(in_features, out_features):
        permuted_out = (torch.arange(out_features) + 1).remainder(out_features)

        weight = torch.zeros(out_features, in_features)
        for i in range(min(out_features, in_features)):
            weight[i, i] = 1.0
        weight = weight[permuted_out, :]

        return weight

    def _assert_input(self, x):
        assert(len(x.size()) == 2)
        assert(x.size(1) == self.in_features)

    def forward(self, x):
        self._assert_input(x)
        return x.matmul(self.weight.t())


layers = []

startRl = ResLinear(2, 99)
startHi = ResSin(99)
layers.append(startRl)
layers.append(startHi)

for _ in range(40):
    layers.append(ResLinear(99, 99))
    layers.append(ResSin(99))
# midRl = ResLinear(99, 99)
# midHi = ResSin(99)
# layers.append(midRl)
# layers.append(midHi)

stopRl = ResLinear(99, 2)
layers.append(stopRl)

parameters = sum(
    [list(layer.parameters()) for layer in layers],
    []
)
optimizer = torch.optim.SGD(
    parameters,
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-4,
)

for i in itertools.count():
    out = data

    for l in layers:
        out = l(out)

    loss = \
        (out[:, 0] - x).square().mean() + \
        (out[:, 1] - y).square().mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 0.1)
    optimizer.step()

    if i % 10 == 0:
        # print(layers[0]._weight())
        # print(layers[2]._weight())

        plt.axis((-3.0, 3.0, -3.0, 3.0))
        plt.scatter(x, y, s=0.01, color='k', alpha=0.25)
        plt.scatter(out.data[:, 0], out.data[:, 1], s=0.5, color='b')
        plt.savefig('figures/out-%05d.png' % i)
        plt.close()

        print(i, loss.item())
