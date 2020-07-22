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

# x = 1.5 * (data[:, 0] * 2).tanh()
# y = 1.5 * (data[:, 0] + data[:, 1]).tanh()

# x = data[:, 0]
# y = (5.0 * data[:, 1] - 5.0).sigmoid() + (5.0 * data[:, 1] + 5.0).sigmoid()


class ResSin(torch.nn.Module):
    def __init__(self, out_features):
        super(ResSin, self).__init__()

        self.bendiness = torch.nn.Parameter(torch.zeros(out_features))

        self.fix_bendy_bias = torch.nn.Parameter(
            torch.randn(out_features),
            requires_grad=False,
        )
        self.var_bendy_bias = torch.nn.Parameter(
            torch.zeros(out_features),
        )

        self.var_straight_bias = torch.nn.Parameter(
            torch.zeros(out_features),
        )

        self.tau = (torch.acos(torch.tensor(0.0)) * 4.0).item()

        self.out_features = out_features

    def forward(self, x):
        straight_bias = \
            self.var_straight_bias

        bendy_bias = \
            self.fix_bendy_bias + \
            self.var_bendy_bias

        straight_bit = \
            x + \
            straight_bias

        bendy_bit = \
            self.bendiness * \
            (x * self.tau + bendy_bias).sin()

        return straight_bit + bendy_bit


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
        permuted_out = torch.randperm(out_features)

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

startRl = ResLinear(2, 222)
startHi = ResSin(222)
layers.append(startRl)
layers.append(startHi)

# for _ in range(40):
#     layers.append(ResLinear(222, 222))
#     layers.append(ResSin(222))
midRl = ResLinear(222, 222)
midHi = ResSin(222)
layers.append(midRl)
layers.append(midHi)

stopRl = ResLinear(222, 2)
layers.append(stopRl)

parameters = sum(
    [list(layer.parameters()) for layer in layers],
    []
)
optimizer = torch.optim.SGD(
    parameters,
    lr=1e-4,
    momentum=0.9,
    weight_decay=1e-4,
)

for i in itertools.count():
    out = data

    out = startRl(out)
    out = startHi(out)

    for _ in range(200):
        out = midRl(out)
        out = midHi(out)

    out = stopRl(out)

    loss = \
        (out[:, 0] - x).square().mean() + \
        (out[:, 1] - y).square().mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1e0)
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
