from dataclasses import dataclass
from warp_core import WarpCore
import torch
from torch import nn
import torchvision
import numpy as np
import json

import sys
import os
import io

from gdf import DDPMSampler, AdaptiveLossWeight

from warp_core.data import MultiGetter

from train_b import WurstCore as WurstCoreBase


class WurstCore(WurstCoreBase, WarpCore):
    @dataclass(frozen=True)
    class ConfigDTO(WurstCoreBase.ConfigDTO):
        bucketeer_random_ratio: float = 0.0

    def webdataset_preprocessors(self, extras: WurstCoreBase.ExtrasDTO):
        def identity(x):
            if isinstance(x, bytes):
                x = x.decode('utf-8')
            return x

        def load_tensor(x):
            return torch.load(io.BytesIO(x), map_location='cpu')

        # CUSTOM CAPTIONS GETTER -----
        def get_caption(oc, c, p_og=0.05): # cog_contexual, cog_caption
            if p_og > 0 and np.random.rand() < p_og and len(oc) > 0:
                return identity(oc)
            else:
                return identity(c)

        captions_getter = MultiGetter(rules={
            ('old_caption', 'caption'): lambda oc, c: get_caption(json.loads(oc)['og_caption'], c, p_og=0.05) 
        })
        # --------

        return [
            ('jpg;png', torchvision.transforms.ToTensor() if self.config.multi_aspect_ratio is not None else extras.transforms, 'images'),
            ('pt', load_tensor, 'latents'),
            ('txt', identity, 'captions') if self.config.captions_getter is None else (self.config.captions_getter[0], eval(self.config.captions_getter[1]), 'captions'),
        ]

    # Extras: gdf, transforms and preprocessors --------------------------------
    def setup_extras_pre(self) -> WurstCoreBase.ExtrasDTO:
        super_extras = super().setup_extras_pre()
        sampling_configs = {"cfg": 1.5, "sampler": DDPMSampler(super_extras.gdf), "shift": 1, "timesteps": 20}

        return self.ExtrasDTO.from_dict({
            **super_extras.to_dict(),
            'sampling_configs': sampling_configs  
        })

    # Data --------------------------------
    def get_conditions(self, batch: dict, models: WurstCoreBase.ModelsDTO, extras: WurstCoreBase.ExtrasDTO, is_eval=False, is_unconditional=False, eval_image_embeds=False, return_fields=None):
        indices = np.random.randint(0, batch['latents'].size(1), size=(batch['latents'].size(0), 1))
        random_embeddings = torch.cat([l[indices[i]] for i, l in enumerate(batch['latents'])], dim=0)

        if is_eval and not is_unconditional:
            effnet_embeddings = random_embeddings
        else:
            effnet_embeddings = torch.zeros_like(random_embeddings)
            rand_idx = np.random.rand(len(effnet_embeddings)) <= 0.9
            if any(rand_idx):
                indices = np.random.randint(0, batch['latents'].size(1), size=(batch['latents'].size(0), 1))
                effnet_embeddings[rand_idx] = random_embeddings[rand_idx]

        conditions = super().get_conditions(
            batch, models, extras, is_eval, is_unconditional,
            eval_image_embeds, return_fields=return_fields or ['clip_text_pooled']
        )

        effnet_embeddings = effnet_embeddings.to(self.device)
        return {'effnet': effnet_embeddings, 'clip': conditions['clip']}

    # Training loop --------------------------------
    def forward_pass(self, data: WarpCore.DataDTO, extras: WurstCoreBase.ExtrasDTO, models: WurstCoreBase.ModelsDTO):
        batch = next(data.iterator)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            epsilon = torch.randn_like(latents)
            epsilon = self._pyramid_noise(epsilon, size_range=[1, 16])
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1, epsilon=epsilon)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)

        return loss, loss_adjusted

if __name__ == '__main__':
    print("Launching Script")
    warpcore = WurstCore(
        config_file_path=sys.argv[1] if len(sys.argv) > 1 else None,
        device=torch.device(int(os.environ.get("SLURM_LOCALID")))
    )
    # warp_core.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD

    # RUN TRAINING
    warpcore()
