import torch
import numpy as np

# --- Loss Weighting
class BaseLossWeight():
    def weight(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1e9, 1e9] if clamp_range is None else clamp_range
        if shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(shift)
        return self.weight(logSNR, *args, **kwargs).clamp(*clamp_range)

class ComposedLossWeight(BaseLossWeight):
    def __init__(self, div, mul):
        self.mul = [mul] if isinstance(mul, BaseLossWeight) else mul
        self.div = [div] if isinstance(div, BaseLossWeight) else div

    def weight(self, logSNR):
        prod, div = 1, 1
        for m in self.mul:
            prod *= m.weight(logSNR)
        for d in self.div:
            div *= d.weight(logSNR)
        return prod/div

class ConstantLossWeight(BaseLossWeight):
    def __init__(self, v=1):
        self.v = v

    def weight(self, logSNR):
        return torch.ones_like(logSNR) * self.v

class SNRLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp()

class P2LossWeight(BaseLossWeight):
    def __init__(self, k=1.0, gamma=1.0, s=1.0):
        self.k, self.gamma, self.s = k, gamma, s

    def weight(self, logSNR):
        return (self.k + (logSNR * self.s).exp()) ** -self.gamma

class SNRPlusOneLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp() + 1

class MinSNRLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr

    def weight(self, logSNR):
        return logSNR.exp().clamp(max=self.max_snr)

class MinSNRPlusOneLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr

    def weight(self, logSNR):
        return (logSNR.exp() + 1).clamp(max=self.max_snr)

class TruncatedSNRLossWeight(BaseLossWeight):
    def __init__(self, min_snr=1):
        self.min_snr = min_snr

    def weight(self, logSNR):
        return logSNR.exp().clamp(min=self.min_snr)

class SechLossWeight(BaseLossWeight):
    def __init__(self, div=2):
        self.div = div

    def weight(self, logSNR):
        return 1/(logSNR/self.div).cosh()

class DebiasedLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return 1/logSNR.exp().sqrt()

class SigmoidLossWeight(BaseLossWeight):
    def __init__(self, s=1):
        self.s = s

    def weight(self, logSNR):
        return (logSNR * self.s).sigmoid()

class AdaptiveLossWeight(BaseLossWeight):
    def __init__(self, logsnr_range=[-10, 10], buckets=300, weight_range=[1e-7, 1e7]):
        self.bucket_ranges = torch.linspace(logsnr_range[0], logsnr_range[1], buckets-1)
        self.bucket_losses = torch.ones(buckets)
        self.weight_range = weight_range

    def weight(self, logSNR):
        indices = torch.searchsorted(self.bucket_ranges.to(logSNR.device), logSNR)
        return (1/self.bucket_losses.to(logSNR.device)[indices]).clamp(*self.weight_range)

    def update_buckets(self, logSNR, loss, beta=0.99):
        indices = torch.searchsorted(self.bucket_ranges.to(logSNR.device), logSNR).cpu()
        self.bucket_losses[indices] = self.bucket_losses[indices]*beta + loss.detach().cpu() * (1-beta)
