import torch
import torch.nn as nn

class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th, slope):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        ctx.slope = slope
        return (v >= v_th).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        v_diff = v - ctx.v_th
        grad_input = grad_output / (1.0 + ctx.slope * torch.abs(v_diff)).pow(2)
        return grad_input, None, None

class LIFCell(nn.Module):
    def __init__(self, beta=0.85, threshold=1.0, slope=25.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.slope = slope

    def forward(self, current, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(current)
        v = self.beta * membrane + current
        spike = FastSigmoidSurrogate.apply(v, self.threshold, self.slope)
        v_reset = v - spike * self.threshold
        return spike, v_reset
