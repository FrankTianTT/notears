import torch
import numpy as np
import torch.nn as nn
import math


class EnsembleLocallyConnected(nn.Module):
    def __init__(self, ensemble_size, num_linear, input_features, output_features, bias=True):
        super(EnsembleLocallyConnected, self).__init__()
        # 这些linear是并行的
        self.ensemble_size = ensemble_size
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(ensemble_size,
                                                num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size,
                                                  num_linear,
                                                  output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # 初始化权重
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # input_numpy: sample-size * node-num * input-features
        # weight: ensemble-size * node-num * input-feature * output-features
        input = input.unsqueeze(dim=1)
        input = input.repeat(1, self.ensemble_size, 1, 1)
        out = torch.einsum("abcd,bcde->abce", input, self.weight)

        if self.bias is not None:
            # [n, e, d, m2] += [e, d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'ensemble_size={}, num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_size, self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )

def calculate_output(input_numpy, weight, ensemble_size):
    # input_numpy: sample-size * node-num * input-features
    # weight: ensemble-size * node-num * input-feature * output-features
    input_numpy = np.expand_dims(input_numpy, axis=1)
    input_numpy = np.tile(input_numpy, [1, ensemble_size, 1, 1])

    # output_numpy: ensemble-size * node-num * output-features
    output_numpy = np.einsum("abcd,bcde->abce", input_numpy, weight)
    return output_numpy


def main():
    # n：随机sample的样本数
    # d：DAG的节点数
    # m1：input features
    # m2: output features
    # 不考虑d的话，这就是一个标准的mlp
    n, d, m1, m2 = 200, 3, 128, 256

    ensemble_size = 7

    input_numpy = np.random.randn(n, d, m1)
    weight = np.random.randn(ensemble_size, d, m1, m2)
    output_numpy = calculate_output(input_numpy.copy(), weight.copy(), ensemble_size)

    # torch
    torch.set_default_dtype(torch.double)
    input_torch = torch.from_numpy(input_numpy)
    locally_connected = EnsembleLocallyConnected(ensemble_size, d, m1, m2, bias=False)
    locally_connected.weight.data[:] = torch.from_numpy(weight)
    output_torch = locally_connected(input_torch)

    # compare
    print(torch.allclose(output_torch, torch.from_numpy(output_numpy)))


if __name__ == '__main__':
    main()
