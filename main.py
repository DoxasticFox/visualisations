#!/usr/bin/env python3

import itertools
import matplotlib.pyplot as plt
import torch

data = torch.randn(1000, 2)
#data = (torch.randn(1000, 2) + 0.5).sin()


class Sin(torch.nn.Module):
    def __init__(self, out_features):
        super(Sin, self).__init__()

        self.bendiness = torch.nn.Parameter(torch.zeros(out_features))

        self.fix_bias = torch.nn.Parameter(
            torch.randn(out_features),
            requires_grad=False,
        )
        self.var_bias = torch.nn.Parameter(
            torch.zeros(out_features),
        )

        self.tau = torch.nn.Parameter(
            torch.acos(torch.tensor(0.0)) * 4.0,
            requires_grad=False,
        )

        self.out_features = out_features

    def forward(self, x):
        bias = self.fix_bias + self.var_bias
        return x + self.bendiness * (x * self.tau + bias).sin()


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

startRl = ResLinear(2, 33)
startHi = Sin(33)
layers.append(startRl)
layers.append(startHi)

midRl = ResLinear(33, 33)
midHi = Sin(33)
layers.append(midRl)
layers.append(midHi)

stopRl = ResLinear(33, 2)
layers.append(stopRl)

parameters = sum(
    [list(layer.parameters()) for layer in layers],
    []
)
optimizer = torch.optim.SGD(
    parameters,
    lr=1e-3,
    momentum=0.9,
    weight_decay=1e-2,
)

for i in itertools.count():
    out = data

    out = startRl(out)
    out = startHi(out)

    for _ in range(1000):
        out = midRl(out)
        out = midHi(out)

    out = stopRl(out)


    # loss = \
    #     (out[:, 0] - data[:, 0].round()).square().mean() + \
    #     (out[:, 1] - data[:, 1]).square().mean()

    div = (data[:, 0].square() + data[:, 1].square()).sqrt()
    x = data[:, 0] / div
    y = data[:, 1] / div
    loss = (out[:, 0] - x).square().mean() + (out[:, 1] - y).square().mean()

    # loss = \
    #     (out[:, 0] - data[:, 0]).square().mean() + \
    #     (out[:, 1] - (data[:, 0] * 3.0).sin()).square().mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 0.1)
    optimizer.step()

    if i % 10 == 0:
        # print(layers[0]._weight())
        # print(layers[2]._weight())

        plt.axis((-3.0, 3.0, -3.0, 3.0))
        plt.scatter(
            out.data[:, 0],
            out.data[:, 1],
            s=0.5,
        )
        plt.savefig('figures/out-%05d.png' % i)
        plt.close()

        print(i, loss.item())
