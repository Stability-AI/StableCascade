import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from warmup_scheduler import GradualWarmupScheduler

import sys
import os
import wandb
from dataclasses import dataclass

from gdf import GDF, EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight
from torchtools.transforms import SmartCrop

from modules import EfficientNetEncoder
from modules import StageC
from modules import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules import Previewer
from modules import ControlNet, ControlNetDeliverer
from modules import controlnet_filters

from train.base import DataCore, TrainingCore

from core import WarpCore
from core.utils import EXPECTED, EXPECTED_TRAIN, load_or_fail

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from contextlib import contextmanager


class WurstCore(TrainingCore, DataCore, WarpCore):
    @dataclass(frozen=True)
    class Config(TrainingCore.Config, DataCore.Config, WarpCore.Config):
        # TRAINING PARAMS
        lr: float = EXPECTED_TRAIN
        warmup_updates: int = EXPECTED_TRAIN
        offset_noise: float = None
        dtype: str = None

        # MODEL VERSION
        model_version: str = EXPECTED  # 3.6B or 1B
        clip_image_model_name: str = 'openai/clip-vit-large-patch14'
        clip_text_model_name: str = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'

        # CHECKPOINT PATHS
        effnet_checkpoint_path: str = EXPECTED
        previewer_checkpoint_path: str = EXPECTED
        generator_checkpoint_path: str = None
        controlnet_checkpoint_path: str = None

        # controlnet settings
        controlnet_blocks: list = EXPECTED
        controlnet_filter: str = EXPECTED
        controlnet_filter_params: dict = None
        controlnet_bottleneck_mode: str = None

    @dataclass(frozen=True)
    class Models(TrainingCore.Models, DataCore.Models, WarpCore.Models):
        effnet: nn.Module = EXPECTED
        previewer: nn.Module = EXPECTED
        controlnet: nn.Module = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        controlnet: any = None

    @dataclass(frozen=True)
    class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: torchvision.transforms.Compose = EXPECTED
        controlnet_filter: controlnet_filters.BaseFilter = EXPECTED

    # @dataclass() # not frozen, means that fields are mutable. Doesn't support EXPECTED
    # class Info(WarpCore.Info):
    #     ema_loss: float = None

    @dataclass(frozen=True)
    class Optimizers(TrainingCore.Optimizers, WarpCore.Optimizers):
        generator: any = None
        controlnet: any = EXPECTED

    info: TrainingCore.Info
    config: Config

    def setup_extras_pre(self) -> Extras:
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(), target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=P2LossWeight(),
            offset_noise=self.config.offset_noise if self.config.offset_noise is not None else 0.0
        )
        sampling_configs = {"cfg": 5, "sampler": DDPMSampler(gdf), "shift": 1, "timesteps": 20}

        effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        if self.config.training:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.config.image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                SmartCrop(self.config.image_size, randomize_p=0.3, randomize_q=0.2)
            ])
        else:
            transforms = None

        controlnet_filter = getattr(controlnet_filters, self.config.controlnet_filter)(
            self.device,
            **(self.config.controlnet_filter_params if self.config.controlnet_filter_params is not None else {})
        )

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess,
            controlnet_filter=controlnet_filter
        )

    def get_cnet(self, batch: dict, models: Models, extras: Extras, cnet_input=None, **kwargs):
        images = batch['images']
        with torch.no_grad():
            if cnet_input is None:
                cnet_input = extras.controlnet_filter(images, **kwargs)
            if isinstance(cnet_input, tuple):
                cnet_input, cnet_input_preview = cnet_input
            else:
                cnet_input_preview = cnet_input
            cnet_input, cnet_input_preview = cnet_input.to(self.device), cnet_input_preview.to(self.device)
        cnet = models.controlnet(cnet_input)
        return cnet, cnet_input_preview

    def get_conditions(self, batch: dict, models: Models, extras: Extras, is_eval=False, is_unconditional=False,
                       eval_image_embeds=False, return_fields=None):
        with torch.no_grad():
            conditions = super().get_conditions(
                batch, models, extras, is_eval, is_unconditional,
                eval_image_embeds, return_fields=return_fields or ['clip_text', 'clip_text_pooled', 'clip_img']
            )
        return conditions

    def setup_models(self, extras: Extras) -> Models:
        dtype = getattr(torch, self.config.dtype) if self.config.dtype else torch.float32

        # EfficientNet encoder
        effnet = EfficientNetEncoder().to(self.device)
        effnet_checkpoint = load_or_fail(self.config.effnet_checkpoint_path)
        effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
        effnet.eval().requires_grad_(False)
        del effnet_checkpoint

        # Previewer
        previewer = Previewer().to(self.device)
        previewer_checkpoint = load_or_fail(self.config.previewer_checkpoint_path)
        previewer.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
        previewer.eval().requires_grad_(False)
        del previewer_checkpoint

        @contextmanager
        def dummy_context():
            yield None

        loading_context = dummy_context if self.config.training else init_empty_weights

        with loading_context():
            # Diffusion models
            if self.config.model_version == '3.6B':
                generator = StageC()
            elif self.config.model_version == '1B':
                generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
            else:
                raise ValueError(f"Unknown model version {self.config.model_version}")
                
        if self.config.generator_checkpoint_path is not None:
            if loading_context is dummy_context:
                generator.load_state_dict(load_or_fail(self.config.generator_checkpoint_path))
            else:
                for param_name, param in load_or_fail(self.config.generator_checkpoint_path).items():
                    set_module_tensor_to_device(generator, param_name, "cpu", value=param)
        generator = generator.to(dtype).to(self.device)
        generator = self.load_model(generator, 'generator')

        # if self.config.use_fsdp:
        #     fsdp_auto_wrap_policy = ModuleWrapPolicy([ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock])
        #     generator = FSDP(generator, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        # CLIP encoders
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(self.config.clip_text_model_name).requires_grad_(False).to(dtype).to(self.device)
        image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name).requires_grad_(False).to(dtype).to(self.device)

        # ControlNet
        controlnet = ControlNet(
            c_in=extras.controlnet_filter.num_channels(),
            proj_blocks=self.config.controlnet_blocks,
            bottleneck_mode=self.config.controlnet_bottleneck_mode
        )

        if self.config.controlnet_checkpoint_path is not None:
            controlnet_checkpoint = load_or_fail(self.config.controlnet_checkpoint_path)
            controlnet.load_state_dict(controlnet_checkpoint if 'state_dict' not in controlnet_checkpoint else controlnet_checkpoint['state_dict'])
        controlnet = controlnet.to(dtype).to(self.device)

        controlnet = self.load_model(controlnet, 'controlnet')
        controlnet.backbone.eval().requires_grad_(True)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=3000)
            controlnet = FSDP(controlnet, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        return self.Models(
            effnet=effnet, previewer=previewer,
            generator=generator, generator_ema=None,
            controlnet=controlnet,
            tokenizer=tokenizer, text_model=text_model, image_model=image_model
        )

    def setup_optimizers(self, extras: Extras, models: Models) -> Optimizers:
        optimizer = optim.AdamW(models.controlnet.parameters(), lr=self.config.lr)  # , eps=1e-7, betas=(0.9, 0.95))
        optimizer = self.load_optimizer(optimizer, 'controlnet_optim',
                                        fsdp_model=models.controlnet if self.config.use_fsdp else None)
        return self.Optimizers(generator=None, controlnet=optimizer)

    def setup_schedulers(self, extras: Extras, models: Models, optimizers: Optimizers) -> Schedulers:
        scheduler = GradualWarmupScheduler(optimizers.controlnet, multiplier=1, total_epoch=self.config.warmup_updates)
        scheduler.last_epoch = self.info.total_steps
        return self.Schedulers(controlnet=scheduler)

    def forward_pass(self, data: WarpCore.Data, extras: Extras, models: Models):
        batch = next(data.iterator)

        cnet, _ = self.get_cnet(batch, models, extras)
        conditions = {**self.get_conditions(batch, models, extras), 'cnet': cnet}
        with torch.no_grad():
            latents = self.encode_latents(batch, models, extras)
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        return loss, loss_adjusted

    def backward_pass(self, update, loss, loss_adjusted, models: Models, optimizers: Optimizers,
                      schedulers: Schedulers):
        if update:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(models.controlnet.parameters(), 1.0)
            optimizers_dict = optimizers.to_dict()
            for k in optimizers_dict:
                if optimizers_dict[k] is not None and k != 'training':
                    optimizers_dict[k].step()
            schedulers_dict = schedulers.to_dict()
            for k in schedulers_dict:
                if k != 'training':
                    schedulers_dict[k].step()
            for k in optimizers_dict:
                if optimizers_dict[k] is not None and k != 'training':
                    optimizers_dict[k].zero_grad(set_to_none=True)
            self.info.total_steps += 1
        else:
            loss_adjusted.backward()
            grad_norm = torch.tensor(0.0).to(self.device)

        return grad_norm

    def models_to_save(self):
        return ['controlnet']  # ['generator', 'generator_ema']

    # LATENT ENCODING & PROCESSING ----------
    def encode_latents(self, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        images = batch['images'].to(self.device)
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        return models.previewer(latents)

    def sample(self, models: Models, data: WarpCore.Data, extras: Extras):
        models.controlnet.eval()
        with torch.no_grad():
            batch = next(data.iterator)

            cnet, cnet_input = self.get_cnet(batch, models, extras)
            conditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
            unconditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)
            conditions, unconditions = {**conditions, 'cnet': cnet}, {**unconditions, 'cnet': cnet}

            latents = self.encode_latents(batch, models, extras)
            noised, _, _, logSNR, noise_cond, _ = extras.gdf.diffuse(latents, shift=1, loss_shift=1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = models.generator(noised, noise_cond, **conditions)
                pred = extras.gdf.undiffuse(noised, logSNR, pred)[0]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                *_, (sampled, _, _) = extras.gdf.sample(
                    models.generator, conditions,
                    latents.shape, unconditions, device=self.device, **extras.sampling_configs
                )

                if models.generator_ema is not None:
                    *_, (sampled_ema, _, _) = extras.gdf.sample(
                        models.generator_ema, conditions,
                        latents.shape, unconditions, device=self.device, **extras.sampling_configs
                    )
                else:
                    sampled_ema = sampled

            if self.is_main_node:
                noised_images = torch.cat(
                    [self.decode_latents(noised[i:i + 1], batch, models, extras) for i in range(len(noised))], dim=0)
                pred_images = torch.cat(
                    [self.decode_latents(pred[i:i + 1], batch, models, extras) for i in range(len(pred))], dim=0)
                sampled_images = torch.cat(
                    [self.decode_latents(sampled[i:i + 1], batch, models, extras) for i in range(len(sampled))], dim=0)
                sampled_images_ema = torch.cat(
                    [self.decode_latents(sampled_ema[i:i + 1], batch, models, extras) for i in range(len(sampled_ema))],
                    dim=0)

                images = batch['images']
                if images.size(-1) != noised_images.size(-1) or images.size(-2) != noised_images.size(-2):
                    images = nn.functional.interpolate(images, size=noised_images.shape[-2:], mode='bicubic')
                    cnet_input = nn.functional.interpolate(cnet_input, size=noised_images.shape[-2:], mode='bicubic')
                    if cnet_input.size(1) == 1:
                        cnet_input = cnet_input.repeat(1, 3, 1, 1)
                    elif cnet_input.size(1) > 3:
                        cnet_input = cnet_input[:, :3]

                collage_img = torch.cat([
                    torch.cat([i for i in images.cpu()], dim=-1),
                    torch.cat([i for i in cnet_input.cpu()], dim=-1),
                    torch.cat([i for i in noised_images.cpu()], dim=-1),
                    torch.cat([i for i in pred_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images_ema.cpu()], dim=-1),
                ], dim=-2)

                torchvision.utils.save_image(collage_img, f'{self.config.output_path}/{self.config.experiment_id}/{self.info.total_steps:06d}.jpg')
                torchvision.utils.save_image(collage_img, f'{self.config.experiment_id}_latest_output.jpg')

                captions = batch['captions']
                if self.config.wandb_project is not None:
                    log_data = [
                        [captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_ema[i])] + [
                            wandb.Image(cnet_input[i])] + [wandb.Image(images[i])] for i in range(len(images))]
                    log_table = wandb.Table(data=log_data,
                                            columns=["Captions", "Sampled", "Sampled EMA", "Cnet", "Orig"])
                    wandb.log({"Log": log_table})
            models.controlnet.train()
            models.controlnet.backbone.eval()


if __name__ == '__main__':
    print("Launching Script")
    warpcore = WurstCore(
        config_file_path=sys.argv[1] if len(sys.argv) > 1 else None,
        device=torch.device(int(os.environ.get("SLURM_LOCALID")))
    )
    warpcore.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD

    # RUN TRAINING
    warpcore()
