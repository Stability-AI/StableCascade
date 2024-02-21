import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torchvision
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchtools.transforms import SmartCrop
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from warmup_scheduler import GradualWarmupScheduler

from core import WarpCore
from core.utils import EXPECTED, EXPECTED_TRAIN, load_or_fail
from gdf import (
    GDF,
    AdaptiveLossWeight,
    CosineSchedule,
    CosineTNoiseCond,
    DDPMSampler,
    EpsilonTarget,
    P2LossWeight,
    VPScaler,
)
from modules.effnet import EfficientNetEncoder
from modules.previewer import Previewer
from modules.stage_c import AttnBlock, FeedForwardBlock, ResBlock, StageC, TimestepBlock
from train.base import DataCore, TrainingCore


class WurstCore(TrainingCore, DataCore, WarpCore):
    @dataclass(frozen=True)
    class Config(TrainingCore.Config, DataCore.Config, WarpCore.Config):
        # TRAINING PARAMS
        lr: float = EXPECTED_TRAIN
        warmup_updates: int = EXPECTED_TRAIN
        dtype: str = None

        # MODEL VERSION
        model_version: str = EXPECTED  # 3.6B or 1B
        clip_image_model_name: str = "openai/clip-vit-large-patch14"
        clip_text_model_name: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

        # CHECKPOINT PATHS
        effnet_checkpoint_path: str = EXPECTED
        previewer_checkpoint_path: str = EXPECTED
        generator_checkpoint_path: str = None

        # gdf customization
        adaptive_loss_weight: str = None

    @dataclass(frozen=True)
    class Models(TrainingCore.Models, DataCore.Models, WarpCore.Models):
        effnet: nn.Module = EXPECTED
        previewer: nn.Module = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        generator: any = None

    @dataclass(frozen=True)
    class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: torchvision.transforms.Compose = EXPECTED

    info: TrainingCore.Info
    config: Config

    def setup_extras_pre(self) -> Extras:
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(),
            target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight()
            if self.config.adaptive_loss_weight is True
            else P2LossWeight(),
        )
        sampling_configs = {
            "cfg": 5,
            "sampler": DDPMSampler(gdf),
            "shift": 1,
            "timesteps": 20,
        }

        if self.info.adaptive_loss is not None:
            gdf.loss_weight.bucket_ranges = torch.tensor(
                self.info.adaptive_loss["bucket_ranges"]
            )
            gdf.loss_weight.bucket_losses = torch.tensor(
                self.info.adaptive_loss["bucket_losses"]
            )

        effnet_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
            ]
        )

        clip_preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                ),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        if self.config.training:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(
                        self.config.image_size,
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    SmartCrop(self.config.image_size, randomize_p=0.3, randomize_q=0.2),
                ]
            )
        else:
            transforms = None

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess,
        )

    def get_conditions(
        self,
        batch: dict,
        models: Models,
        extras: Extras,
        is_eval=False,
        is_unconditional=False,
        eval_image_embeds=False,
        return_fields=None,
    ):
        return super().get_conditions(
            batch,
            models,
            extras,
            is_eval,
            is_unconditional,
            eval_image_embeds,
            return_fields=return_fields
            or ["clip_text", "clip_text_pooled", "clip_img"],
        )

    def setup_models(self, extras: Extras) -> Models:
        dtype = (
            getattr(torch, self.config.dtype) if self.config.dtype else torch.float32
        )

        # EfficientNet encoder
        effnet = EfficientNetEncoder()
        effnet_checkpoint = load_or_fail(self.config.effnet_checkpoint_path)
        effnet.load_state_dict(
            effnet_checkpoint
            if "state_dict" not in effnet_checkpoint
            else effnet_checkpoint["state_dict"]
        )
        effnet.eval().requires_grad_(False).to(self.device)
        del effnet_checkpoint

        # Previewer
        previewer = Previewer()
        previewer_checkpoint = load_or_fail(self.config.previewer_checkpoint_path)
        previewer.load_state_dict(
            previewer_checkpoint
            if "state_dict" not in previewer_checkpoint
            else previewer_checkpoint["state_dict"]
        )
        previewer.eval().requires_grad_(False).to(self.device)
        del previewer_checkpoint

        @contextmanager
        def dummy_context():
            yield None

        loading_context = dummy_context if self.config.training else init_empty_weights

        # Diffusion models
        with loading_context():
            generator_ema = None
            if self.config.model_version == "3.6B":
                generator = StageC()
                if self.config.ema_start_iters is not None:
                    generator_ema = StageC()
            elif self.config.model_version == "1B":
                generator = StageC(
                    c_cond=1536,
                    c_hidden=[1536, 1536],
                    nhead=[24, 24],
                    blocks=[[4, 12], [12, 4]],
                )
                if self.config.ema_start_iters is not None:
                    generator_ema = StageC(
                        c_cond=1536,
                        c_hidden=[1536, 1536],
                        nhead=[24, 24],
                        blocks=[[4, 12], [12, 4]],
                    )
            else:
                raise ValueError(f"Unknown model version {self.config.model_version}")

        if self.config.generator_checkpoint_path is not None:
            if loading_context is dummy_context:
                generator.load_state_dict(
                    load_or_fail(self.config.generator_checkpoint_path)
                )
            else:
                for param_name, param in load_or_fail(
                    self.config.generator_checkpoint_path
                ).items():
                    set_module_tensor_to_device(
                        generator, param_name, "cpu", value=param
                    )
        generator = generator.to(dtype).to(self.device)
        generator = self.load_model(generator, "generator")

        if generator_ema is not None:
            if loading_context is dummy_context:
                generator_ema.load_state_dict(generator.state_dict())
            else:
                for param_name, param in generator.state_dict().items():
                    set_module_tensor_to_device(
                        generator_ema, param_name, "cpu", value=param
                    )
            generator_ema = self.load_model(generator_ema, "generator_ema")
            generator_ema.to(dtype).to(self.device).eval().requires_grad_(False)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = ModuleWrapPolicy(
                [ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock]
            )
            generator = FSDP(
                generator,
                **self.fsdp_defaults,
                auto_wrap_policy=fsdp_auto_wrap_policy,
                device_id=self.device,
            )
            if generator_ema is not None:
                generator_ema = FSDP(
                    generator_ema,
                    **self.fsdp_defaults,
                    auto_wrap_policy=fsdp_auto_wrap_policy,
                    device_id=self.device,
                )

        # CLIP encoders
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = (
            CLIPTextModelWithProjection.from_pretrained(
                self.config.clip_text_model_name
            )
            .requires_grad_(False)
            .to(dtype)
            .to(self.device)
        )
        image_model = (
            CLIPVisionModelWithProjection.from_pretrained(
                self.config.clip_image_model_name
            )
            .requires_grad_(False)
            .to(dtype)
            .to(self.device)
        )

        return self.Models(
            effnet=effnet,
            previewer=previewer,
            generator=generator,
            generator_ema=generator_ema,
            tokenizer=tokenizer,
            text_model=text_model,
            image_model=image_model,
        )

    def setup_optimizers(
        self, extras: Extras, models: Models
    ) -> TrainingCore.Optimizers:
        optimizer = optim.AdamW(
            models.generator.parameters(), lr=self.config.lr
        )  # , eps=1e-7, betas=(0.9, 0.95))
        optimizer = self.load_optimizer(
            optimizer,
            "generator_optim",
            fsdp_model=models.generator if self.config.use_fsdp else None,
        )
        return self.Optimizers(generator=optimizer)

    def setup_schedulers(
        self, extras: Extras, models: Models, optimizers: TrainingCore.Optimizers
    ) -> Schedulers:
        scheduler = GradualWarmupScheduler(
            optimizers.generator, multiplier=1, total_epoch=self.config.warmup_updates
        )
        scheduler.last_epoch = self.info.total_steps
        return self.Schedulers(generator=scheduler)

    # Training loop --------------------------------
    def forward_pass(self, data: WarpCore.Data, extras: Extras, models: Models):
        batch = next(data.iterator)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(
                latents, shift=1, loss_shift=1
            )

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            loss = nn.functional.mse_loss(pred, target, reduction="none").mean(
                dim=[1, 2, 3]
            )
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)

        return loss, loss_adjusted

    def backward_pass(
        self,
        update,
        loss,
        loss_adjusted,
        models: Models,
        optimizers: TrainingCore.Optimizers,
        schedulers: Schedulers,
    ):
        if update:
            return self._extracted_from_backward_pass_11(
                loss_adjusted, models, optimizers, schedulers
            )
        loss_adjusted.backward()
        return torch.tensor(0.0).to(self.device)

    # TODO Rename this here and in `backward_pass`
    def _extracted_from_backward_pass_11(
        self, loss_adjusted, models, optimizers, schedulers
    ):
        loss_adjusted.backward()
        result = nn.utils.clip_grad_norm_(models.generator.parameters(), 1.0)
        optimizers_dict = optimizers.to_dict()
        for k in optimizers_dict:
            if k != "training":
                optimizers_dict[k].step()
        schedulers_dict = schedulers.to_dict()
        for k in schedulers_dict:
            if k != "training":
                schedulers_dict[k].step()
        for k in optimizers_dict:
            if k != "training":
                optimizers_dict[k].zero_grad(set_to_none=True)
        self.info.total_steps += 1
        return result

    def models_to_save(self):
        return ["generator", "generator_ema"]

    def encode_latents(
        self, batch: dict, models: Models, extras: Extras
    ) -> torch.Tensor:
        images = batch["images"].to(self.device)
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(
        self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras
    ) -> torch.Tensor:
        return models.previewer(latents)


if __name__ == "__main__":
    print("Launching Script")
    warpcore = WurstCore(
        config_file_path=sys.argv[1] if len(sys.argv) > 1 else None,
        device=torch.device(int(os.environ.get("SLURM_LOCALID"))),
    )
    # core.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD

    # RUN TRAINING
    warpcore()
