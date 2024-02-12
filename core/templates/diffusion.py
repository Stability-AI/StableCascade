from .. import WarpCore
from ..utils import EXPECTED, EXPECTED_TRAIN, update_weights_ema, create_folder_if_necessary
from abc import abstractmethod
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from gdf import GDF
import numpy as np
from tqdm import tqdm
import wandb

import webdataset as wds
from webdataset.handlers import warn_and_continue
from torch.distributed import barrier
from enum import Enum

class TargetReparametrization(Enum):
    EPSILON = 'epsilon'
    X0 = 'x0'

class DiffusionCore(WarpCore):
    @dataclass(frozen=True)
    class Config(WarpCore.Config):
        # TRAINING PARAMS
        lr: float = EXPECTED_TRAIN
        grad_accum_steps: int = EXPECTED_TRAIN
        batch_size: int = EXPECTED_TRAIN
        updates: int = EXPECTED_TRAIN
        warmup_updates: int = EXPECTED_TRAIN
        save_every: int = 500
        backup_every: int = 20000
        use_fsdp: bool = True

        # EMA UPDATE
        ema_start_iters: int = None
        ema_iters: int = None
        ema_beta: float = None

        # GDF setting
        gdf_target_reparametrization: TargetReparametrization = None # epsilon or x0
    
    @dataclass() # not frozen, means that fields are mutable. Doesn't support EXPECTED
    class Info(WarpCore.Info):
        ema_loss: float = None

    @dataclass(frozen=True)
    class Models(WarpCore.Models):
        generator : nn.Module = EXPECTED
        generator_ema : nn.Module = None # optional

    @dataclass(frozen=True)
    class Optimizers(WarpCore.Optimizers):
        generator : any = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        generator: any = None

    @dataclass(frozen=True)
    class Extras(WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED

    # --------------------------------------------
    info: Info
    config: Config

    @abstractmethod
    def encode_latents(self, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def get_conditions(self, batch: dict, models: Models, extras: Extras, is_eval=False, is_unconditional=False):
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def webdataset_path(self, extras: Extras):
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def webdataset_filters(self, extras: Extras):
        raise NotImplementedError("This method needs to be overriden")
    
    @abstractmethod
    def webdataset_preprocessors(self, extras: Extras):
        raise NotImplementedError("This method needs to be overriden")

    @abstractmethod
    def sample(self, models: Models, data: WarpCore.Data, extras: Extras):
        raise NotImplementedError("This method needs to be overriden")
    # -------------

    def setup_data(self, extras: Extras) -> WarpCore.Data:
        # SETUP DATASET
        dataset_path = self.webdataset_path(extras)
        preprocessors = self.webdataset_preprocessors(extras)
        filters = self.webdataset_filters(extras)

        handler = warn_and_continue # None
        # handler = None
        dataset = wds.WebDataset(
            dataset_path, resampled=True, handler=handler
        ).select(filters).shuffle(690, handler=handler).decode(
            "pilrgb", handler=handler
        ).to_tuple(
            *[p[0] for p in preprocessors], handler=handler
        ).map_tuple(
            *[p[1] for p in preprocessors], handler=handler
        ).map(lambda x: {p[2]:x[i] for i, p in enumerate(preprocessors)})

        # SETUP DATALOADER
        real_batch_size = self.config.batch_size//(self.world_size*self.config.grad_accum_steps)
        dataloader = DataLoader(
            dataset, batch_size=real_batch_size, num_workers=8, pin_memory=True
        )

        return self.Data(dataset=dataset, dataloader=dataloader, iterator=iter(dataloader))

    def forward_pass(self, data: WarpCore.Data, extras: Extras, models: Models):
        batch = next(data.iterator)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)

        # FORWARD PASS
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            if self.config.gdf_target_reparametrization == TargetReparametrization.EPSILON:
                pred = extras.gdf.undiffuse(noised, logSNR, pred)[1] # transform whatever prediction to epsilon to use in the loss
                target = noise
            elif self.config.gdf_target_reparametrization == TargetReparametrization.X0:
                pred = extras.gdf.undiffuse(noised, logSNR, pred)[0] # transform whatever prediction to x0 to use in the loss
                target = latents
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        return loss, loss_adjusted
    
    def train(self, data: WarpCore.Data, extras: Extras, models: Models, optimizers: Optimizers, schedulers: Schedulers):
        start_iter = self.info.iter+1
        max_iters = self.config.updates * self.config.grad_accum_steps
        if self.is_main_node:
            print(f"STARTING AT STEP: {start_iter}/{max_iters}")

        pbar = tqdm(range(start_iter, max_iters+1)) if self.is_main_node else range(start_iter, max_iters+1) # <--- DDP
        models.generator.train()
        for i in pbar:
            # FORWARD PASS
            loss, loss_adjusted = self.forward_pass(data, extras, models)

            # BACKWARD PASS
            if i % self.config.grad_accum_steps == 0 or i == max_iters:
                loss_adjusted.backward()
                grad_norm = nn.utils.clip_grad_norm_(models.generator.parameters(), 1.0)
                optimizers_dict = optimizers.to_dict()
                for k in optimizers_dict:
                    optimizers_dict[k].step()
                schedulers_dict = schedulers.to_dict()
                for k in schedulers_dict:
                    schedulers_dict[k].step()
                models.generator.zero_grad(set_to_none=True)
                self.info.total_steps += 1
            else:
                with models.generator.no_sync():
                    loss_adjusted.backward()
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
                    'lr': optimizers.generator.param_groups[0]['lr'],
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
                        wandb.alert(title=f"Skipping sampling & checkpoint for training run {self.config.run_id}", text=f"Skipping sampling & checkpoint at {self.info.total_steps} for training run {self.info.wandb_run_id} iters because loss is NaN")
                else:
                    self.save_checkpoints(models, optimizers)
                    if self.is_main_node:
                        create_folder_if_necessary(f'{self.config.output_path}/{self.config.experiment_id}/')
                    self.sample(models, data, extras)

    def models_to_save(self):
        return ['generator', 'generator_ema']

    def save_checkpoints(self, models: Models, optimizers: Optimizers, suffix=None):
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
                self.save_optimizer(optimizer, f'{key}_optim{suffix}', fsdp_model=models.generator if self.config.use_fsdp else None)
        if suffix == '' and self.info.total_steps > 1 and self.info.total_steps % self.config.backup_every == 0:
            self.save_checkpoints(models, optimizers, suffix=f"_{self.info.total_steps//1000}k")
        torch.cuda.empty_cache()
