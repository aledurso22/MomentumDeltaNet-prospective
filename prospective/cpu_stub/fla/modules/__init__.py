import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu', **kw):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         padding=kernel_size - 1, bias=bias)
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kw):
        T = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), self.weight, self.bias,
                     padding=self.kernel_size[0] - 1, groups=self.groups)[..., :T]
        return F.silu(y.transpose(1, 2)), None


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation='swish'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps, self.activation = eps, activation

    def forward(self, x, g):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return y * (torch.sigmoid(g) if self.activation == 'sigmoid' else F.silu(g))


RMSNorm = L2Norm = None
