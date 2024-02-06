import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, layer, name='weight', rank=16, alpha=1):
        super().__init__()
        weight = getattr(layer, name)
        self.lora_down = nn.Parameter(torch.zeros((rank, weight.size(1))))
        self.lora_up = nn.Parameter(torch.zeros((weight.size(0), rank)))
        nn.init.normal_(self.lora_up, mean=0, std=1)

        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            lora_shape = list(original_weights.shape[:2]) + [1] * (len(original_weights.shape) - 2)
            lora_weights = torch.matmul(self.lora_up.clone(), self.lora_down.clone()).view(*lora_shape) * self.scale
            return original_weights + lora_weights
        else:
            return original_weights


def apply_lora(model, filters=None, rank=16):
    def check_parameter(module, name):
        return hasattr(module, name) and not torch.nn.utils.parametrize.is_parametrized(module, name) and isinstance(
            getattr(module, name), nn.Parameter)

    for name, module in model.named_modules():
        if filters is None or any([f in name for f in filters]):
            if check_parameter(module, "weight"):
                torch.nn.utils.parametrize.register_parametrization(module, 'weight', LoRA(module, "weight", rank=rank))
            elif check_parameter(module, "in_proj_weight"):
                torch.nn.utils.parametrize.register_parametrization(module, 'in_proj_weight',
                                                                    LoRA(module, "in_proj_weight", rank=rank))


class ReToken(nn.Module):
    def __init__(self, indices=None):
        super().__init__()
        assert indices is not None
        self.embeddings = nn.Parameter(torch.zeros(len(indices), 1280))
        self.register_buffer('indices', torch.tensor(indices))
        self.enabled = True

    def forward(self, embeddings):
        if self.enabled:
            embeddings = embeddings.clone()
            for i, idx in enumerate(self.indices):
                embeddings[idx] += self.embeddings[i]
        return embeddings


def apply_retoken(module, indices=None):
    def check_parameter(module, name):
        return hasattr(module, name) and not torch.nn.utils.parametrize.is_parametrized(module, name) and isinstance(
            getattr(module, name), nn.Parameter)

    if check_parameter(module, "weight"):
        device = module.weight.device
        torch.nn.utils.parametrize.register_parametrization(module, 'weight', ReToken(indices=indices).to(device))


def remove_lora(model, leave_parametrized=True):
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=leave_parametrized)
        elif torch.nn.utils.parametrize.is_parametrized(module, "in_proj_weight"):
            nn.utils.parametrize.remove_parametrizations(module, "in_proj_weight",
                                                         leave_parametrized=leave_parametrized)
