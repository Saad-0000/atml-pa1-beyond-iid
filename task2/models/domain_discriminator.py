import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator used by DANN and CDAN.

    DANN:
        512 -> 256 -> 2

    CDAN:
        3584 -> 256 -> 2

    Dropout is retained as specified by the assignment architecture.
    """

    def __init__(self, in_features=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)