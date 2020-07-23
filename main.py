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

        tau = (torch.acos(torch.tensor(0.0)) * 4.0).item()
        w_min = - 2.0 * tau
        w_max =   2.0 * tau

        self.inner_weight = torch.nn.Parameter(
            ResSin._range_non_zero(out_features, w_min, w_max),
            requires_grad=False,
        )
        self.outer_weight = torch.nn.Parameter(
            ResSin._range_two(
                (out_features,),
                -1, -w_max, w_max, 1
            ) / self.inner_weight,
            requires_grad=False,
        )

        self.out_features = out_features

    def forward(self, x):
        bendy_bit = \
            self.bendiness * \
            self.outer_weight * \
            (x * self.inner_weight).sin()

        return x + bendy_bit

    @staticmethod
    def _range(num, lo, hi):
        assert(num >= 2)
        return torch.arange(num) / float(num - 1) * (hi - lo) + lo

    @staticmethod
    def _bisect(nums):
        if len(nums) == 1:
            l_num = nums[0] // 2
            r_num = nums[0] - l_num
            return l_num, r_num
        elif len(nums) == 2:
            return nums
        else:
            ValueError('Too many nums')

    @staticmethod
    def _range_two(nums, l_lo, l_hi, r_lo, r_hi):
        l_num, r_num = ResSin._bisect(nums)

        return torch.cat((
            ResSin._range(l_num, l_lo, l_hi),
            ResSin._range(r_num, r_lo, r_hi),
        ))

    @staticmethod
    def _range_non_zero(num, lo, hi):
        l_num, r_num = ResSin._bisect((num,))

        l_lo = lo
        l_hi = -1.0 / l_num
        r_lo =  1.0 / r_num
        r_hi = hi

        return ResSin._range_two((l_num, r_num), l_lo, l_hi, r_lo, r_hi)


class ResLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ResLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            ResLinear._initial_weight(in_features, out_features)
        )

        self.bias = torch.nn.Parameter(
            torch.zeros(out_features)
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

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight.t())


layers = []

startRl = ResLinear(2, 344)
startHi = ResSin(344)
layers.append(startRl)
layers.append(startHi)

# for _ in range(40):
#     layers.append(ResLinear(344, 344))
#     layers.append(ResSin(344))
midRl = ResLinear(344, 344)
midHi = ResSin(344)
layers.append(midRl)
layers.append(midHi)

stopRl = ResLinear(344, 2)
layers.append(stopRl)

parameters = sum(
    [list(layer.parameters()) for layer in layers],
    []
)
optimizer = torch.optim.SGD(
    parameters,
    lr=2e-5,
    momentum=0.9,
    #weight_decay=1e-3,
)

for i in itertools.count():
    out = data

    out = startRl(out)
    out = startHi(out)

    for _ in range(1000):
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
