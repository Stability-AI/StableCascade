from warp_core import WarpCore
import torch
from torch import nn
from transformers import AutoTokenizer, CLIPModel, CLIPVisionModelWithProjection

import sys
import os

from modules.effnet import EfficientNetEncoder
from modules.stage_c import StageC
from modules.stage_c import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules.previewer import Previewer

from train_c import WurstCore as WurstCoreBase

from gdf import AdaptiveLossWeight
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

class WurstCore(WurstCoreBase, WarpCore):
    # Models, Optimizers & Schedulers setup --------------------------------
    def setup_models(self, extras: WurstCoreBase.ExtrasDTO) -> WurstCoreBase.ModelsDTO:
        # EfficientNet encoder
        effnet = EfficientNetEncoder().to(self.device)
        effnet_checkpoint = torch.load(self.config.effnet_checkpoint_path, map_location=self.device)
        effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
        effnet.eval().requires_grad_(False)
        del effnet_checkpoint

        # Previewer
        previewer = Previewer().to(self.device)
        previewer_checkpoint = torch.load(self.config.previewer_checkpoint_path, map_location=self.device)
        previewer.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
        previewer.eval().requires_grad_(False)
        del previewer_checkpoint

        # Diffusion models 
        if self.config.model_version == '3.6B':
            generator = StageC(switch_level=[True]).to(self.device) # <---------- HERE'S THE ONLY CHANGE TO THE ORIGINAL CODE: 'switch_level=[True]'
            if self.config.ema_start_iters is not None:
                generator_ema = StageC(switch_level=[True]).to(self.device) # <---------- HERE'S THE ONLY CHANGE TO THE ORIGINAL CODE: 'switch_level=[True]'
            else:
                generator_ema = None
        elif self.config.model_version == '1B':
            generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]], switch_level=[True]).to(self.device)  # <---------- HERE'S THE ONLY CHANGE TO THE ORIGINAL CODE: 'switch_level=[True]'
            if self.config.ema_start_iters is not None:
                generator_ema = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]], switch_level=[True]).to(self.device) # <---------- HERE'S THE ONLY CHANGE TO THE ORIGINAL CODE: 'switch_level=[True]'
            else:
                generator_ema = None
        else:
            raise ValueError(f"Unknown model version {self.config.model_version}")

        if self.config.generator_checkpoint_path is not None:
            generator.load_state_dict(torch.load(self.config.generator_checkpoint_path, map_location=self.device))
        generator = self.load_model(generator, 'generator')

        if generator_ema is not None:
            generator_ema.load_state_dict(generator.state_dict())
            generator_ema = self.load_model(generator_ema, 'generator_ema')
            generator_ema.eval().requires_grad_(False)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = ModuleWrapPolicy([ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock])
            generator = FSDP(generator, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)
            if generator_ema is not None:
                generator_ema = FSDP(generator_ema, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        # CLIP encoders
        clip_tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        clip_model = CLIPModel.from_pretrained(self.config.clip_text_model_name)
        clip_text_model = clip_model.text_model.to(self.device).eval().requires_grad_(False)
        clip_text_model_proj = clip_model.text_projection.to(self.device).eval().requires_grad_(False)
        clip_image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name).to(self.device).eval().requires_grad_(False)
        del clip_model

        return self.ModelsDTO(
            effnet=effnet, previewer=previewer,
            generator=generator, generator_ema=generator_ema,

            clip_tokenizer=clip_tokenizer, clip_text_model=clip_text_model,
            clip_text_model_proj=clip_text_model_proj, clip_image_model=clip_image_model
        )

    # Training loop --------------------------------
    def forward_pass(self, data: WarpCore.DataDTO, extras: WurstCoreBase.ExtrasDTO, models: WurstCoreBase.ModelsDTO):
        batch = next(data.iterator)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=2, loss_shift=1)

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
