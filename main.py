#!/usr/bin/env python3

import itertools
import matplotlib.pyplot as plt
import torch

data = torch.randn(1000, 2)
#data = (torch.randn(1000, 2) + 0.5).sin()


class Hinge(torch.nn.Module):
    def __init__(self, out_features):
        super(Hinge, self).__init__()

        self.hi = torch.nn.Parameter(torch.zeros(out_features))
        self.lo = torch.nn.Parameter(torch.zeros(out_features))

        self.var_bias = torch.nn.Parameter(torch.zeros(out_features))
        self.register_buffer('fix_bias', torch.randn(out_features))

        self.out_features = out_features

    def _bias(self):
        return self.var_bias + self.fix_bias

    def forward(self, x):
        shifted_x = x + self._bias()
        bendy_x = (
            shifted_x.clamp(min=0) * (1 - self.hi) +
            shifted_x.clamp(max=0) * (1 - self.lo)
        )
        return bendy_x - self._bias()


class ResLinear(torch.nn.Module):
    weights = {}

    def __init__(self, in_features, out_features):
        super(ResLinear, self).__init__()

        self.var_weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features)
        )
        self.register_buffer(
            'fix_weight',
            self._fix_weight(out_features, in_features)
        )

        self.in_features = in_features
        self.out_features = out_features

    def _weight(self):
        return self.var_weight + self.fix_weight

    def _fix_weight(self, out_features, in_features):
        key = (out_features, in_features)
        rkey = tuple(reversed(key))

        if rkey in ResLinear.weights:
            return ResLinear.weights[rkey].t()

        if out_features == in_features and False:
            return torch.eye(out_features)
        else:
            ResLinear.weights[key] = torch.empty(out_features, in_features)
            torch.nn.init.orthogonal_(ResLinear.weights[key])
            return ResLinear.weights[key]


        # rand = torch.randn(out_features, in_features)
        # rand = rand / rand.norm(dim=1, keepdim=True)
        # return rand



        square_size = min(out_features, in_features)

        eye_part = torch.eye(square_size)

        if square_size < out_features:
            rand_part = torch.randn(out_features - square_size, square_size)
            rand_part = rand_part / rand_part.norm(dim=1, keepdim=True)

            fix_weight = torch.cat((eye_part, rand_part), dim=0)
        elif square_size < in_features:
            zero_part = torch.zeros(square_size, in_features - square_size)

            fix_weight = torch.cat((eye_part, zero_part), dim=1)
        else:
            fix_weight = eye_part

        assert(fix_weight.size() == (out_features, in_features))
        return fix_weight

    def forward(self, x):
        return x.matmul(self._weight().t())

layers = []

layers.append(ResLinear(2, 10))
layers.append(Hinge(10))

for _ in range(10):
    layers.append(ResLinear(10, 10))

    layers.append(Hinge(10))

layers.append(ResLinear(10, 2))


parameters = sum(
    [list(layer.parameters()) for layer in layers],
    []
)
optimizer = torch.optim.SGD(
    parameters,
    lr=1e-2,
    momentum=0.9,
    #weight_decay=1e-3,
)

print(layers[0]._weight())
print(layers[2]._weight())
for i in itertools.count():
    out = data
    for layer in layers:
        out = layer(out)

    # loss = (out[:, 0] - data[:, 0].round()).square().mean()

    div = (data[:, 0].square() + data[:, 1].square()).sqrt()
    x = data[:, 0] / div
    y = data[:, 1] / div
    loss = (out[:, 0] - x).square().mean() + (out[:, 1] - y).square().mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 0.1)
    optimizer.step()

    if i % 100 == 0:
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
