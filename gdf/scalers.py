import torch

class BaseScaler():
    def __init__(self):
        self.stretched_limits = None

    def setup_limits(self, schedule, input_scaler, stretch_max=True, stretch_min=True, shift=1):
        min_logSNR = schedule(torch.ones(1), shift=shift)
        max_logSNR = schedule(torch.zeros(1), shift=shift)

        min_a, max_b = [v.item() for v in input_scaler(min_logSNR)] if stretch_max else [0, 1]
        max_a, min_b = [v.item() for v in input_scaler(max_logSNR)] if stretch_min else [1, 0]
        self.stretched_limits = [min_a, max_a, min_b, max_b]
        return self.stretched_limits

    def stretch_limits(self, a, b):
        min_a, max_a, min_b, max_b = self.stretched_limits
        return (a - min_a) / (max_a - min_a), (b - min_b) / (max_b - min_b)

    def scalers(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR):
        a, b = self.scalers(logSNR)
        if self.stretched_limits is not None:
            a, b = self.stretch_limits(a, b)
        return a, b

class VPScaler(BaseScaler):
    def scalers(self, logSNR):
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1-a_squared).sqrt()
        return a, b

class LERPScaler(BaseScaler):
    def scalers(self, logSNR):
        _a = logSNR.exp() - 1
        _a[_a == 0] = 1e-3 # Avoid division by zero
        a = 1 + (2-(2**2 + 4*_a)**0.5) / (2*_a)
        b = 1-a
        return a, b
