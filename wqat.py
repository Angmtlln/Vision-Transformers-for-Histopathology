import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearWQAT(nn.Module):
    def __init__(self, base: nn.Linear, per_channel: bool = True):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.has_bias = base.bias is not None

        self.weight = nn.Parameter(base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone()) if self.has_bias else None

        from torch.ao.quantization import FakeQuantize, default_per_channel_weight_observer, default_weight_observer
        if per_channel:
            self.weight_fake = FakeQuantize(
                observer=default_per_channel_weight_observer,
                quant_min=-128, quant_max=127,
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
            )
        else:
            self.weight_fake = FakeQuantize(
                observer=default_weight_observer,
                quant_min=-128, quant_max=127,
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            )

    def forward(self, x):
        w_q = self.weight_fake(self.weight)
        return F.linear(x, w_q, self.bias)