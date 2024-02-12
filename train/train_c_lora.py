import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from warmup_scheduler import GradualWarmupScheduler

import sys
import os
import re
from dataclasses import dataclass

from gdf import GDF, EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from torchtools.transforms import SmartCrop

from modules.effnet import EfficientNetEncoder
from modules.stage_c import StageC
from modules.stage_c import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules.previewer import Previewer
from modules.lora import apply_lora, apply_retoken, LoRA, ReToken

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
        dtype: str = None

        # MODEL VERSION
        model_version: str = EXPECTED  # 3.6B or 1B
        clip_image_model_name: str = 'openai/clip-vit-large-patch14'
        clip_text_model_name: str = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'

        # CHECKPOINT PATHS
        effnet_checkpoint_path: str = EXPECTED
        previewer_checkpoint_path: str = EXPECTED
        generator_checkpoint_path: str = None
        lora_checkpoint_path: str = None

        # LoRA STUFF
        module_filters: list = EXPECTED
        rank: int = EXPECTED
        train_tokens: list = EXPECTED

        # gdf customization
        adaptive_loss_weight: str = None

    @dataclass(frozen=True)
    class Models(TrainingCore.Models, DataCore.Models, WarpCore.Models):
        effnet: nn.Module = EXPECTED
        previewer: nn.Module = EXPECTED
        lora: nn.Module = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        lora: any = None

    @dataclass(frozen=True)
    class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: torchvision.transforms.Compose = EXPECTED

    @dataclass()  # not frozen, means that fields are mutable. Doesn't support EXPECTED
    class Info(TrainingCore.Info):
        train_tokens: list = None

    @dataclass(frozen=True)
    class Optimizers(TrainingCore.Optimizers, WarpCore.Optimizers):
        generator: any = None
        lora: any = EXPECTED

    # --------------------------------------------
    info: Info
    config: Config

    # Extras: gdf, transforms and preprocessors --------------------------------
    def setup_extras_pre(self) -> Extras:
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(), target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight() if self.config.adaptive_loss_weight is True else P2LossWeight(),
        )
        sampling_configs = {"cfg": 5, "sampler": DDPMSampler(gdf), "shift": 1, "timesteps": 20}

        if self.info.adaptive_loss is not None:
            gdf.loss_weight.bucket_ranges = torch.tensor(self.info.adaptive_loss['bucket_ranges'])
            gdf.loss_weight.bucket_losses = torch.tensor(self.info.adaptive_loss['bucket_losses'])

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

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess
        )

    # Data --------------------------------
    def get_conditions(self, batch: dict, models: Models, extras: Extras, is_eval=False, is_unconditional=False,
                       eval_image_embeds=False, return_fields=None):
        conditions = super().get_conditions(
            batch, models, extras, is_eval, is_unconditional,
            eval_image_embeds, return_fields=return_fields or ['clip_text', 'clip_text_pooled', 'clip_img']
        )
        return conditions

    # Models, Optimizers & Schedulers setup --------------------------------
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
        #     fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=3000)
        #     generator = FSDP(generator, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        # CLIP encoders
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(self.config.clip_text_model_name).requires_grad_(False).to(dtype).to(self.device)
        image_model = CLIPVisionModelWithProjection.from_pretrained(self.config.clip_image_model_name).requires_grad_(False).to(dtype).to(self.device)

        # PREPARE LORA
        update_tokens = []
        for tkn_regex, aggr_regex in self.config.train_tokens:
            if (tkn_regex.startswith('[') and tkn_regex.endswith(']')) or (tkn_regex.startswith('<') and tkn_regex.endswith('>')):
                # Insert new token
                tokenizer.add_tokens([tkn_regex])
                # add new zeros embedding
                new_embedding = torch.zeros_like(text_model.text_model.embeddings.token_embedding.weight.data)[:1]
                if aggr_regex is not None:  # aggregate embeddings to provide an interesting baseline
                    aggr_tokens = [v for k, v in tokenizer.vocab.items() if re.search(aggr_regex, k) is not None]
                    if len(aggr_tokens) > 0:
                        new_embedding = text_model.text_model.embeddings.token_embedding.weight.data[aggr_tokens].mean(dim=0, keepdim=True)
                    elif self.is_main_node:
                        print(f"WARNING: No tokens found for aggregation regex {aggr_regex}. It will be initialized as zeros.")
                text_model.text_model.embeddings.token_embedding.weight.data = torch.cat([
                    text_model.text_model.embeddings.token_embedding.weight.data, new_embedding
                ], dim=0)
                selected_tokens = [len(tokenizer.vocab) - 1]
            else:
                selected_tokens = [v for k, v in tokenizer.vocab.items() if re.search(tkn_regex, k) is not None]
            update_tokens += selected_tokens
        update_tokens = list(set(update_tokens))  # remove duplicates

        apply_retoken(text_model.text_model.embeddings.token_embedding, update_tokens)
        apply_lora(generator, filters=self.config.module_filters, rank=self.config.rank)
        text_model.text_model.to(self.device)
        generator.to(self.device)
        lora = nn.ModuleDict()
        lora['embeddings'] = text_model.text_model.embeddings.token_embedding.parametrizations.weight[0]
        lora['weights'] = nn.ModuleList()
        for module in generator.modules():
            if isinstance(module, LoRA) or (hasattr(module, '_fsdp_wrapped_module') and isinstance(module._fsdp_wrapped_module, LoRA)):
                lora['weights'].append(module)

        self.info.train_tokens = [(i, tokenizer.decode(i)) for i in update_tokens]
        if self.is_main_node:
            print("Updating tokens:", self.info.train_tokens)
            print(f"LoRA training {len(lora['weights'])} layers")

        if self.config.lora_checkpoint_path is not None:
            lora_checkpoint = load_or_fail(self.config.lora_checkpoint_path)
            lora.load_state_dict(lora_checkpoint if 'state_dict' not in lora_checkpoint else lora_checkpoint['state_dict'])

        lora = self.load_model(lora, 'lora')
        lora.to(self.device).train().requires_grad_(True)
        if self.config.use_fsdp:
            # fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=3000)
            fsdp_auto_wrap_policy = ModuleWrapPolicy([LoRA, ReToken])
            lora = FSDP(lora, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        return self.Models(
            effnet=effnet, previewer=previewer,
            generator=generator, generator_ema=None,
            lora=lora,
            tokenizer=tokenizer, text_model=text_model, image_model=image_model
        )

    def setup_optimizers(self, extras: Extras, models: Models) -> Optimizers:
        optimizer = optim.AdamW(models.lora.parameters(), lr=self.config.lr)  # , eps=1e-7, betas=(0.9, 0.95))
        optimizer = self.load_optimizer(optimizer, 'lora_optim',
                                        fsdp_model=models.lora if self.config.use_fsdp else None)
        return self.Optimizers(generator=None, lora=optimizer)

    def setup_schedulers(self, extras: Extras, models: Models, optimizers: Optimizers) -> Schedulers:
        scheduler = GradualWarmupScheduler(optimizers.lora, multiplier=1, total_epoch=self.config.warmup_updates)
        scheduler.last_epoch = self.info.total_steps
        return self.Schedulers(lora=scheduler)

    def forward_pass(self, data: WarpCore.Data, extras: Extras, models: Models):
        batch = next(data.iterator)

        conditions = self.get_conditions(batch, models, extras)
        with torch.no_grad():
            latents = self.encode_latents(batch, models, extras)
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)

        return loss, loss_adjusted

    def backward_pass(self, update, loss, loss_adjusted, models: Models, optimizers: TrainingCore.Optimizers, schedulers: Schedulers):
        if update:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(models.lora.parameters(), 1.0)
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
        return ['lora']

    def sample(self, models: Models, data: WarpCore.Data, extras: Extras):
        models.lora.eval()
        super().sample(models, data, extras)
        models.lora.train(), models.generator.eval()

    def encode_latents(self, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        images = batch['images'].to(self.device)
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        return models.previewer(latents)


if __name__ == '__main__':
    print("Launching Script")
    warpcore = WurstCore(
        config_file_path=sys.argv[1] if len(sys.argv) > 1 else None,
        device=torch.device(int(os.environ.get("SLURM_LOCALID")))
    )
    warpcore.fsdp_defaults['sharding_strategy'] = ShardingStrategy.NO_SHARD

    # RUN TRAINING
    warpcore()
