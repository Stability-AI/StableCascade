import torch
from torch import nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from warp_core import WarpCore
from warp_core.data import setup_webdataset_path, MultiGetter, MultiFilter
from dataclasses import dataclass
from warp_core.utils import DTO_REQUIRED, update_weights_ema, create_folder_if_necessary
from abc import abstractmethod
from tqdm import tqdm
import wandb
import yaml
import json
from torch.distributed import barrier
from gdf import GDF
from fractions import Fraction
from bucketeer import Bucketeer

import webdataset as wds
from webdataset.handlers import warn_and_continue

# DATA template -----------------------
class DataCore(WarpCore):
    @dataclass(frozen=True)
    class ConfigDTO(WarpCore.ConfigDTO):
        image_size: int = DTO_REQUIRED
        webdataset_path: str = DTO_REQUIRED
        grad_accum_steps: int = DTO_REQUIRED
        batch_size: int = DTO_REQUIRED
        multi_aspect_ratio: list = None

        captions_getter: list = None
        dataset_filters: list = None

    @dataclass(frozen=True)
    class ExtrasDTO(WarpCore.ExtrasDTO):
        transforms: torchvision.transforms.Compose = DTO_REQUIRED
        clip_preprocess: torchvision.transforms.Compose = DTO_REQUIRED

    @dataclass(frozen=True)
    class ModelsDTO(WarpCore.ModelsDTO):
        clip_tokenizer: nn.Module = DTO_REQUIRED
        clip_text_model: nn.Module = DTO_REQUIRED
        clip_text_model_proj: nn.Module = DTO_REQUIRED
        clip_image_model: nn.Module = DTO_REQUIRED

    # --------------------------------------------
    config: ConfigDTO

    def webdataset_path(self):
        if isinstance(self.config.webdataset_path, str) and self.config.webdataset_path.strip().startswith('pipe:'):
            return self.config.webdataset_path
        else:
            dataset_path = self.config.webdataset_path
            if isinstance(self.config.webdataset_path, str) and self.config.webdataset_path.strip().endswith('.yml'):
                with open(self.config.webdataset_path, 'r', encoding='utf-8') as file:
                    dataset_path = yaml.safe_load(file)
            return setup_webdataset_path(dataset_path, cache_path=f"{self.config.experiment_id}_webdataset_cache.yml")

    def webdataset_preprocessors(self, extras: ExtrasDTO):
        def identity(x):
            if isinstance(x, bytes):
                x = x.decode('utf-8')
            return x

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
            ('txt', identity, 'captions') if self.config.captions_getter is None else (self.config.captions_getter[0], eval(self.config.captions_getter[1]), 'captions'),
        ]

    def setup_data(self, extras: ExtrasDTO) -> WarpCore.DataDTO:
        # SETUP DATASET
        dataset_path = self.webdataset_path()
        preprocessors = self.webdataset_preprocessors(extras)

        handler = warn_and_continue
        # handler = None
        dataset = wds.WebDataset(
            dataset_path, resampled=True, handler=handler
        ).select(
            MultiFilter(rules={
                f[0]: eval(f[1]) for f in self.config.dataset_filters
            }) if self.config.dataset_filters is not None else lambda _: True
        ).shuffle(690, handler=handler).decode(
            "pilrgb", handler=handler
        ).to_tuple(
            *[p[0] for p in preprocessors], handler=handler
        ).map_tuple(
            *[p[1] for p in preprocessors], handler=handler
        ).map(lambda x: {p[2]:x[i] for i, p in enumerate(preprocessors)})

        def identity(x):
            return x

        # SETUP DATALOADER
        real_batch_size = self.config.batch_size//(self.world_size*self.config.grad_accum_steps)
        dataloader = DataLoader(
            dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True, 
            collate_fn=identity if self.config.multi_aspect_ratio is not None else None
        )
        if self.is_main_node:
            print(f"Training with batch size {self.config.batch_size} ({real_batch_size}/GPU)")

        if self.config.multi_aspect_ratio is not None:
            aspect_ratios = [float(Fraction(f)) for f in self.config.multi_aspect_ratio]
            dataloader_iterator = Bucketeer(dataloader, density=self.config.image_size**2, factor=32, ratios=aspect_ratios, p_random_ratio=0.05, interpolate_nearest=False) # , use_smartcrop=True)
        else:
            dataloader_iterator = iter(dataloader)

        return self.DataDTO(dataset=dataset, dataloader=dataloader, iterator=dataloader_iterator)

    def get_conditions(self, batch: dict, models: ModelsDTO, extras: ExtrasDTO, is_eval=False, is_unconditional=False, eval_image_embeds=False, return_fields=None):
        if return_fields is None:
            return_fields = ['clip_text', 'clip_text_pooled', 'clip_img']

        captions = batch['captions']
        images = batch['images'].to(self.device)

        if 'clip_text' in return_fields:
            if is_eval:
                if is_unconditional:
                    captions_unpooled = ["" for _ in range(len(images))]
                else:
                    captions_unpooled = captions
            else:
                rand_idx = np.random.rand(len(captions)) > 0.05
                captions_unpooled = [str(c) if keep else "" for c, keep in zip(captions, rand_idx)]
            clip_tokens_unpooled = models.clip_tokenizer(captions_unpooled, truncation=True, padding="max_length", max_length=models.clip_tokenizer.model_max_length, return_tensors="pt").to(self.device)
            clip_text_embeddings = models.clip_text_model(**clip_tokens_unpooled, output_hidden_states=True).hidden_states[-1]
        else:
            clip_text_embeddings = None

        if 'clip_text_pooled' in return_fields:
            if is_eval:
                if is_unconditional:
                    captions_pooled = ["" for _ in range(len(images))]
                else:
                    captions_pooled = captions
            else:
                rand_idx = np.random.rand(len(captions)) > 0.5
                captions_pooled = [str(c) if keep else "" for c, keep in zip(captions, rand_idx)]
            clip_tokens_pooled = models.clip_tokenizer(captions_pooled, truncation=True, padding="max_length", max_length=models.clip_tokenizer.model_max_length, return_tensors="pt").to(self.device)
            clip_text_pooled_embeddings = models.clip_text_model_proj(models.clip_text_model(**clip_tokens_pooled, output_hidden_states=True).pooler_output).unsqueeze(1)
        else:
            clip_text_pooled_embeddings = None

        if 'clip_img' in return_fields:
            clip_image_embeddings = torch.zeros(len(images), 1, 768, device=self.device)
            if is_eval:
                if not is_unconditional and eval_image_embeds:
                    clip_image_embeddings = models.clip_image_model(extras.clip_preprocess(images)).image_embeds
            else:
                rand_idx = np.random.rand(len(images)) > 0.9
                if any(rand_idx):
                    clip_image_embeddings[rand_idx, 0] = models.clip_image_model(extras.clip_preprocess(images[rand_idx])).image_embeds
        else:
            clip_image_embeddings = None

        return {
            'clip_text': clip_text_embeddings, 
            'clip_text_pooled': clip_text_pooled_embeddings, 
            'clip_img': clip_image_embeddings
        }

class TrainingCore(DataCore, WarpCore):
    @dataclass(frozen=True)
    class ConfigDTO(DataCore.ConfigDTO, WarpCore.ConfigDTO):
        updates: int = DTO_REQUIRED
        backup_every: int = DTO_REQUIRED
        save_every: int = DTO_REQUIRED

        # EMA UPDATE
        ema_start_iters: int = None
        ema_iters: int = None
        ema_beta: float = None

        use_fsdp: bool = DTO_REQUIRED

    @dataclass() # not frozen, means that fields are mutable. Doesn't support DTO_REQUIRED
    class InfoDTO(WarpCore.InfoDTO):
        ema_loss: float = None

    @dataclass(frozen=True)
    class ModelsDTO(WarpCore.ModelsDTO):
        generator : nn.Module = DTO_REQUIRED
        generator_ema : nn.Module = None # optional

    @dataclass(frozen=True)
    class OptimizersDTO(WarpCore.OptimizersDTO):
        generator : any = DTO_REQUIRED

    @dataclass(frozen=True)
    class ExtrasDTO(WarpCore.ExtrasDTO):
        gdf: GDF = DTO_REQUIRED
        sampling_configs: dict = DTO_REQUIRED

    # ------
    info: InfoDTO
    config: ConfigDTO

    @abstractmethod
    def forward_pass(self, data: WarpCore.DataDTO, extras: WarpCore.ExtrasDTO, models: ModelsDTO):
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def backward_pass(self, update, loss, loss_adjusted, models: ModelsDTO, optimizers: OptimizersDTO, schedulers: WarpCore.SchedulersDTO):
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def models_to_save(self) -> list:
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def encode_latents(self, batch: dict, models: ModelsDTO, extras: ExtrasDTO) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor, batch: dict, models: ModelsDTO, extras: ExtrasDTO) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overriden")

    # ------

    def train(self, data: WarpCore.DataDTO, extras: WarpCore.ExtrasDTO, models: ModelsDTO, optimizers: OptimizersDTO, schedulers: WarpCore.SchedulersDTO):
        start_iter = self.info.iter+1
        max_iters = self.config.updates * self.config.grad_accum_steps
        if self.is_main_node:
            print(f"STARTING AT STEP: {start_iter}/{max_iters}")

        pbar = tqdm(range(start_iter, max_iters+1)) if self.is_main_node else range(start_iter, max_iters+1) # <--- DDP
        if 'generator' in self.models_to_save():
            models.generator.train()
        for i in pbar:
            # FORWARD PASS
            loss, loss_adjusted = self.forward_pass(data, extras, models)

            # # BACKWARD PASS
            grad_norm = self.backward_pass(
                i % self.config.grad_accum_steps == 0 or i == max_iters, loss, loss_adjusted, 
                models, optimizers, schedulers
            )
            self.info.iter = i

            # UPDATE EMA
            if models.generator_ema is not None and i % self.config.ema_iters == 0:
                update_weights_ema(
                    models.generator_ema, models.generator,
                    beta=(self.config.ema_beta if i > self.config.ema_start_iters else 0)
                )

            # UPDATE LOSS METRICS
            self.info.ema_loss = loss.mean().item() if self.info.ema_loss is None else self.info.ema_loss * 0.99 + loss.mean().item() * 0.01

            if self.is_main_node and self.config.wandb_project is not None and np.isnan(loss.mean().item()) or np.isnan(grad_norm.item()):
                wandb.alert(
                    title=f"NaN value encountered in training run {self.info.wandb_run_id}",
                    text=f"Loss {loss.mean().item()} - Grad Norm {grad_norm.item()}. Run {self.info.wandb_run_id}",
                    wait_duration=60*30
                )

            if self.is_main_node:
                logs = {
                    'loss': self.info.ema_loss, 
                    'raw_loss': loss.mean().item(),
                    'grad_norm': grad_norm.item(),
                    'lr': optimizers.generator.param_groups[0]['lr'] if optimizers.generator is not None else 0,
                    'total_steps': self.info.total_steps,
                }

                pbar.set_postfix(logs)
                if self.config.wandb_project is not None:
                    wandb.log(logs)

            if i == 1 or i % (self.config.save_every*self.config.grad_accum_steps) == 0 or i == max_iters:
                # SAVE AND CHECKPOINT STUFF
                if np.isnan(loss.mean().item()):
                    if self.is_main_node and self.config.wandb_project is not None:
                        tqdm.write("Skipping sampling & checkpoint because the loss is NaN")
                        wandb.alert(title=f"Skipping sampling & checkpoint for training run {self.config.wandb_run_id}", text=f"Skipping sampling & checkpoint at {self.info.total_steps} for training run {self.info.wandb_run_id} iters because loss is NaN")
                else:
                    self.save_checkpoints(models, optimizers)
                    if self.is_main_node:
                        create_folder_if_necessary(f'{self.config.output_path}/{self.config.experiment_id}/')
                    self.sample(models, data, extras)

    def save_checkpoints(self, models: ModelsDTO, optimizers: OptimizersDTO, suffix=None):
        barrier()
        suffix = '' if suffix is None else suffix
        self.save_info(self.info, suffix=suffix)
        models_dict = models.to_dict()
        optimizers_dict = optimizers.to_dict()
        for key in self.models_to_save():
            model = models_dict[key]
            if model is not None:
                self.save_model(model, f"{key}{suffix}", is_fsdp=self.config.use_fsdp)
        for key in optimizers_dict:
            optimizer = optimizers_dict[key]
            if optimizer is not None:
                self.save_optimizer(optimizer, f'{key}_optim{suffix}', fsdp_model=models_dict[key] if self.config.use_fsdp else None)
        if suffix == '' and self.info.total_steps > 1 and self.info.total_steps % self.config.backup_every == 0:
            self.save_checkpoints(models, optimizers, suffix=f"_{self.info.total_steps//1000}k")
        torch.cuda.empty_cache()

    def sample(self, models: ModelsDTO, data: WarpCore.DataDTO, extras: ExtrasDTO):
        if 'generator' in self.models_to_save():
            models.generator.eval()
        with torch.no_grad():
            batch = next(data.iterator)

            conditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False, eval_image_embeds=False)
            unconditions = self.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True, eval_image_embeds=False)

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

                collage_img = torch.cat([
                    torch.cat([i for i in images.cpu()], dim=-1),
                    torch.cat([i for i in noised_images.cpu()], dim=-1),
                    torch.cat([i for i in pred_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images.cpu()], dim=-1),
                    torch.cat([i for i in sampled_images_ema.cpu()], dim=-1),
                ], dim=-2)

                torchvision.utils.save_image(collage_img, f'{self.config.output_path}/{self.config.experiment_id}/{self.info.total_steps:06d}.jpg')
                torchvision.utils.save_image(collage_img, f'{self.config.experiment_id}_latest_output.jpg')

                captions = batch['captions']
                if self.config.wandb_project is not None:
                    log_data = [[captions[i]] + [wandb.Image(sampled_images[i])] + [wandb.Image(sampled_images_ema[i])] + [wandb.Image(images[i])] for i in range(len(images))]
                    log_table = wandb.Table(data=log_data, columns=["Captions", "Sampled", "Sampled EMA", "Orig"])
                    wandb.log({"Log": log_table})
            if 'generator' in self.models_to_save():
                models.generator.train()
