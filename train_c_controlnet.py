from warp_core import WarpCore
from warp_core.utils import DTO_REQUIRED
from dataclasses import dataclass
import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPModel, CLIPVisionModelWithProjection
from warmup_scheduler import GradualWarmupScheduler

import sys
import os
import wandb

from gdf import GDF, EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight
from torchtools.transforms import SmartCrop

from modules.effnet import EfficientNetEncoder
from modules.stage_c import StageC
from modules.stage_c import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules.previewer import Previewer
from modules.controlnet import ControlNet, ControlNetDeliverer
import modules.controlnet as controlnet_filters

from train_templates import DataCore, TrainingCore

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

class WurstCore(TrainingCore, DataCore, WarpCore):
    # DTOs ---------------------------------------
    @dataclass(frozen=True)
    class ConfigDTO(TrainingCore.ConfigDTO, DataCore.ConfigDTO, WarpCore.ConfigDTO):
        # TRAINING PARAMS
        lr: float = DTO_REQUIRED
        warmup_updates: int = DTO_REQUIRED
        offset_noise: float = None

        # MODEL VERSION
        model_version: str = DTO_REQUIRED # 3.6B or 1B
        clip_image_model_name: str = 'openai/clip-vit-large-patch14'
        clip_text_model_name: str = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'

        # CHECKPOINT PATHS
        effnet_checkpoint_path: str = DTO_REQUIRED
        previewer_checkpoint_path: str = DTO_REQUIRED
        generator_checkpoint_path: str = None

        # controlnet settings
        controlnet_blocks: list = DTO_REQUIRED
        controlnet_filter: str = DTO_REQUIRED
        controlnet_filter_params: dict = None
        controlnet_skip_effnet: bool = None

    @dataclass(frozen=True)
    class ModelsDTO(TrainingCore.ModelsDTO, DataCore.ModelsDTO, WarpCore.ModelsDTO):
        effnet: nn.Module = DTO_REQUIRED
        previewer: nn.Module = DTO_REQUIRED
        controlnet: nn.Module = DTO_REQUIRED

    @dataclass(frozen=True)
    class SchedulersDTO(WarpCore.SchedulersDTO):
        controlnet: any = None

    @dataclass(frozen=True)
    class ExtrasDTO(TrainingCore.ExtrasDTO, DataCore.ExtrasDTO, WarpCore.ExtrasDTO):
        gdf: GDF = DTO_REQUIRED
        sampling_configs: dict = DTO_REQUIRED
        effnet_preprocess: torchvision.transforms.Compose = DTO_REQUIRED
        controlnet_filter: controlnet_filters.BaseFilter = DTO_REQUIRED

    # @dataclass() # not frozen, means that fields are mutable. Doesn't support DTO_REQUIRED
    # class InfoDTO(WarpCore.InfoDTO):
    #     ema_loss: float = None

    @dataclass(frozen=True)
    class OptimizersDTO(TrainingCore.OptimizersDTO, WarpCore.OptimizersDTO):
        generator: any = None
        controlnet: any = DTO_REQUIRED

    # --------------------------------------------
    info: TrainingCore.InfoDTO
    config: ConfigDTO

    # Extras: gdf, transforms and preprocessors --------------------------------
    def setup_extras_pre(self) -> ExtrasDTO:
        gdf = GDF(
            schedule = CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler = VPScaler(), target = EpsilonTarget(),
            noise_cond = CosineTNoiseCond(),
            loss_weight = P2LossWeight(),
            offset_noise = self.config.offset_noise if self.config.offset_noise is not None else 0.0
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

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.config.image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
            SmartCrop(self.config.image_size, randomize_p=0.3, randomize_q=0.2)
        ])

        controlnet_filter = getattr(controlnet_filters, self.config.controlnet_filter)(
            self.device, **(self.config.controlnet_filter_params if self.config.controlnet_filter_params is not None else {})
        )

        return self.ExtrasDTO(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess,
            controlnet_filter=controlnet_filter
        )

    # Data --------------------------------
    def get_cnet(self, batch: dict, models: ModelsDTO, extras: ExtrasDTO):
        images = batch['images']
        with torch.no_grad():
            cnet_input = extras.controlnet_filter(images)
            if isinstance(cnet_input, tuple):
                cnet_input, cnet_input_preview = cnet_input
            else:
                cnet_input_preview = cnet_input
            cnet_input, cnet_input_preview = cnet_input.to(self.device), cnet_input_preview.to(self.device)
        cnet = models.controlnet(cnet_input)
        return cnet, cnet_input_preview

    def get_conditions(self, batch: dict, models: ModelsDTO, extras: ExtrasDTO, is_eval=False, is_unconditional=False, eval_image_embeds=False, return_fields=None):
        with torch.no_grad():
            conditions = super().get_conditions(
                batch, models, extras, is_eval, is_unconditional,
                eval_image_embeds, return_fields=return_fields or ['clip_text', 'clip_text_pooled', 'clip_img']
            )
        return conditions

    # Models, Optimizers & Schedulers setup --------------------------------
    def setup_models(self, extras: ExtrasDTO) -> ModelsDTO:
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
            generator = StageC().to(self.device)
        elif self.config.model_version == '1B':
            generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]]).to(self.device)
        else:
            raise ValueError(f"Unknown model version {self.config.model_version}")

        if self.config.generator_checkpoint_path is not None:
            generator.load_state_dict(torch.load(self.config.generator_checkpoint_path, map_location=self.device))
        generator.eval().requires_grad_(False)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = ModuleWrapPolicy([ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock])
            generator = FSDP(generator, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        # CLIP encoders
        clip_tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        clip_model = CLIPModel.from_pretrained(self.config.clip_text_model_name)
        clip_text_model = clip_model.text_model.to(self.device).eval().requires_grad_(False)
        clip_text_model_proj = clip_model.text_projection.to(self.device).eval().requires_grad_(False)
        clip_image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name).to(self.device).eval().requires_grad_(False)
        del clip_model

        # ControlNet
        controlnet = ControlNet(
            c_in=extras.controlnet_filter.num_channels(), 
            proj_blocks=self.config.controlnet_blocks, 
            skip_effnet= self.config.controlnet_skip_effnet if self.config.controlnet_skip_effnet is not None else False
        ).to(self.device)
        controlnet = self.load_model(controlnet, 'controlnet')
        controlnet.backbone.eval().requires_grad_(True)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=3000)
            controlnet = FSDP(controlnet, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        return self.ModelsDTO(
            effnet=effnet, previewer=previewer,
            generator=generator, generator_ema=None,
            controlnet=controlnet,

            clip_tokenizer=clip_tokenizer, clip_text_model=clip_text_model,
            clip_text_model_proj=clip_text_model_proj, clip_image_model=clip_image_model
        )

    def setup_optimizers(self, extras: ExtrasDTO, models: ModelsDTO) -> OptimizersDTO:
        optimizer = optim.AdamW(models.controlnet.parameters(), lr=self.config.lr) #, eps=1e-7, betas=(0.9, 0.95))
        optimizer = self.load_optimizer(optimizer, 'controlnet_optim', fsdp_model=models.controlnet if self.config.use_fsdp else None)
        return self.OptimizersDTO(generator=None, controlnet=optimizer)

    def setup_schedulers(self, extras: ExtrasDTO, models: ModelsDTO, optimizers: OptimizersDTO) -> SchedulersDTO:
        scheduler = GradualWarmupScheduler(optimizers.controlnet, multiplier=1, total_epoch=self.config.warmup_updates)
        scheduler.last_epoch = self.info.total_steps
        return self.SchedulersDTO(controlnet=scheduler)

    # Training loop --------------------------------
    def forward_pass(self, data: WarpCore.DataDTO, extras: ExtrasDTO, models: ModelsDTO):
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

    def backward_pass(self, update, loss, loss_adjusted, models: ModelsDTO, optimizers: OptimizersDTO, schedulers: SchedulersDTO):
        if update:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(models.controlnet.parameters(), 1.0)
            optimizers_dict = optimizers.to_dict()
            for k in optimizers_dict:
                if optimizers_dict[k] is not None:
                    optimizers_dict[k].step()
            schedulers_dict = schedulers.to_dict()
            for k in schedulers_dict:
                schedulers_dict[k].step()
            for k in optimizers_dict:
                if optimizers_dict[k] is not None:
                    optimizers_dict[k].zero_grad(set_to_none=True)
            self.info.total_steps += 1
        else:
            with models.controlnet.no_sync():
                loss_adjusted.backward()

        return grad_norm

    def models_to_save(self):
        return ['controlnet'] # ['generator', 'generator_ema']

    # LATENT ENCODING & PROCESSING ----------
    def encode_latents(self, batch: dict, models: ModelsDTO, extras: ExtrasDTO) -> torch.Tensor:
        images = batch['images'].to(self.device)
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(self, latents: torch.Tensor, batch: dict, models: ModelsDTO, extras: ExtrasDTO) -> torch.Tensor:
        return models.previewer(latents)

    def sample(self, models: ModelsDTO, data: WarpCore.DataDTO, extras: ExtrasDTO):
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
                noised_images = torch.cat([self.decode_latents(noised[i:i+1], batch, models, extras) for i in range(len(noised))], dim=0)
                pred_images = torch.cat([self.decode_latents(pred[i:i+1], batch, models, extras) for i in range(len(pred))], dim=0)
                sampled_images = torch.cat([self.decode_latents(sampled[i:i+1], batch, models, extras) for i in range(len(sampled))], dim=0)
                sampled_images_ema = torch.cat([self.decode_latents(sampled_ema[i:i+1], batch, models, extras) for i in range(len(sampled_ema))], dim=0)

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
                    log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_ema[i])] + [wandb.Image(cnet_input[i])] + [wandb.Image(images[i])] for i in range(len(images))]
                    log_table = wandb.Table(data=log_data, columns=["Captions", "Sampled", "Sampled EMA", "Cnet", "Orig"])
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
