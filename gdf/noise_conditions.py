import torch
import numpy as np

class BaseNoiseCond():
    def __init__(self, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1e9, 1e9] if clamp_range is None else clamp_range
        self.shift = shift
        self.clamp_range = clamp_range
        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass # this method is optional, override it if required

    def cond(self, logSNR):
        raise NotImplementedError("this method needs to be overriden")

    def __call__(self, logSNR):
        if self.shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(self.shift)
        return self.cond(logSNR).clamp(*self.clamp_range)

class CosineTNoiseCond(BaseNoiseCond):
    def setup(self, s=0.008, clamp_range=[0, 1]): # [0.0001, 0.9999]
        self.s = torch.tensor([s])
        self.clamp_range = clamp_range
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def cond(self, logSNR):
        var = logSNR.sigmoid()
        var = var.clamp(*self.clamp_range)
        s, min_var = self.s.to(var.device), self.min_var.to(var.device)
        t = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return t

class EDMNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return -logSNR/8

class SigmoidNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return (-logSNR).sigmoid()

class LogSNRNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return logSNR

class EDMSigmaNoiseCond(BaseNoiseCond):
    def setup(self, sigma_data=1):
        self.sigma_data = sigma_data

    def cond(self, logSNR):
        return torch.exp(-logSNR / 2) * self.sigma_data

class RectifiedFlowsNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        _a = logSNR.exp() - 1
        _a[_a == 0] = 1e-3 # Avoid division by zero
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        return a

# Any NoiseCond that cannot be described easily as a continuous function of t
# It needs to define self.x and self.y in the setup() method
class PiecewiseLinearNoiseCond(BaseNoiseCond):
    def setup(self):
        self.x = None
        self.y = None

    def piecewise_linear(self, y, xs, ys):
        indices = (len(xs)-2) - torch.searchsorted(ys.flip(dims=(-1,))[:-2], y)  
        x_min, x_max = xs[indices], xs[indices+1]
        y_min, y_max = ys[indices], ys[indices+1]
        x = x_min + (x_max - x_min) * (y - y_min) / (y_max - y_min)
        return x

    def cond(self, logSNR):
        var = logSNR.sigmoid()
        t = self.piecewise_linear(var, self.x.to(var.device), self.y.to(var.device)) # .mul(1000).round().clamp(min=0)
        return t

class StableDiffusionNoiseCond(PiecewiseLinearNoiseCond):
    def setup(self, linear_range=[0.00085, 0.012], total_steps=1000):
        self.total_steps = total_steps
        linear_range_sqrt = [r**0.5 for r in linear_range]
        self.x = torch.linspace(0, 1, total_steps+1)

        alphas = 1-(linear_range_sqrt[0]*(1-self.x) + linear_range_sqrt[1]*self.x)**2
        self.y = alphas.cumprod(dim=-1)

    def cond(self, logSNR):
        return super().cond(logSNR).clamp(0, 1)

class DiscreteNoiseCond(BaseNoiseCond):
    def setup(self, noise_cond, steps=1000, continuous_range=[0, 1]):
        self.noise_cond = noise_cond
        self.steps = steps
        self.continuous_range = continuous_range

    def cond(self, logSNR):
        cond = self.noise_cond(logSNR)
        cond = (cond-self.continuous_range[0]) / (self.continuous_range[1]-self.continuous_range[0])
        return cond.mul(self.steps).long()
    