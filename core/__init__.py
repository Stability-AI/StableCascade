import os
import yaml
import torch
from torch import nn
import wandb
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType
)

from .utils import Base, EXPECTED, EXPECTED_TRAIN
from .utils import create_folder_if_necessary, safe_save, load_or_fail

# pylint: disable=unused-argument
class WarpCore(ABC):
    @dataclass(frozen=True)
    class Config(Base):
        experiment_id: str = EXPECTED_TRAIN
        checkpoint_path: str = EXPECTED_TRAIN
        output_path: str = EXPECTED_TRAIN
        checkpoint_extension: str = "safetensors"
        dist_file_subfolder: str = ""
        allow_tf32: bool = True

        wandb_project: str = None
        wandb_entity: str = None

    @dataclass() # not frozen, means that fields are mutable
    class Info(): # not inheriting from Base, because we don't want to enforce the default fields
        wandb_run_id: str = None
        total_steps: int = 0
        iter: int = 0

    @dataclass(frozen=True)
    class Data(Base):
        dataset: Dataset = EXPECTED
        dataloader: DataLoader  = EXPECTED
        iterator: any = EXPECTED

    @dataclass(frozen=True)
    class Models(Base):
        pass

    @dataclass(frozen=True)
    class Optimizers(Base):
        pass

    @dataclass(frozen=True)
    class Schedulers(Base):
        pass

    @dataclass(frozen=True)
    class Extras(Base):
        pass
    # ---------------------------------------
    info: Info
    config: Config

    # FSDP stuff
    fsdp_defaults = {
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
        "cpu_offload": None,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        "limit_all_gathers": True,
    }
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    # ------------

    # OVERRIDEABLE METHODS
    
    # [optionally] setup extra stuff, will be called BEFORE the models & optimizers are setup
    def setup_extras_pre(self) -> Extras:
        return self.Extras()

    # setup dataset & dataloader, return a dict contained dataser, dataloader and/or iterator
    @abstractmethod
    def setup_data(self, extras: Extras) -> Data:
        raise NotImplementedError("This method needs to be overriden")

    # return a dict with all models that are going to be used in the training
    @abstractmethod
    def setup_models(self, extras: Extras) -> Models:
        raise NotImplementedError("This method needs to be overriden")

    # return a dict with all optimizers that are going to be used in the training
    @abstractmethod
    def setup_optimizers(self, extras: Extras, models: Models) -> Optimizers:
        raise NotImplementedError("This method needs to be overriden")

    # [optionally] return a dict with all schedulers that are going to be used in the training
    def setup_schedulers(self, extras: Extras, models: Models, optimizers: Optimizers) -> Schedulers:
        return self.Schedulers()

    # [optionally] setup extra stuff, will be called AFTER the models & optimizers are setup
    def setup_extras_post(self, extras: Extras, models: Models, optimizers: Optimizers, schedulers: Schedulers) -> Extras:
        return self.Extras.from_dict(extras.to_dict())

    # perform the training here
    @abstractmethod
    def train(self, data: Data, extras: Extras, models: Models, optimizers: Optimizers, schedulers: Schedulers, single_gpu: bool=False):
        raise NotImplementedError("This method needs to be overriden")
    # ------------

    def setup_info(self, full_path=None) -> Info:
        if full_path is None:
            full_path = (f"{self.config.checkpoint_path}/{self.config.experiment_id}/info.json")
        info_dict = load_or_fail(full_path, wandb_run_id=None) or {}
        info_dto = self.Info(**info_dict)
        if info_dto.total_steps > 0 and self.is_main_node:
            print(">>> RESUMING TRAINING FROM ITER ", info_dto.total_steps)
        return info_dto

    def setup_config(self, config_file_path=None, config_dict=None, training=True) -> Config:
        if config_file_path is not None:
            if config_file_path.endswith(".yml") or config_file_path.endswith(".yaml"):
                with open(config_file_path, "r", encoding="utf-8") as file:
                    loaded_config = yaml.safe_load(file)
            elif config_file_path.endswith(".json"):
                with open(config_file_path, "r", encoding="utf-8") as file:
                    loaded_config = json.load(file)
            else:
                raise ValueError("Config file must be either a .yml|.yaml or .json file")
            return self.Config.from_dict({**loaded_config, 'training': training})
        if config_dict is not None:
            return self.Config.from_dict({**config_dict, 'training': training})
        return self.Config(training=training)

    def setup_ddp(self, experiment_id, single_gpu=False):
        if not single_gpu:
            local_rank = int(os.environ.get("SLURM_LOCALID"))
            process_id = int(os.environ.get("SLURM_PROCID"))
            world_size = int(os.environ.get("SLURM_NNODES")) * torch.cuda.device_count()

            self.process_id = process_id
            self.is_main_node = process_id == 0
            self.device = torch.device(local_rank)
            self.world_size = world_size

            dist_file_path = f"{os.getcwd()}/{self.config.dist_file_subfolder}dist_file_{experiment_id}"
            # if os.path.exists(dist_file_path) and self.is_main_node:
            #     os.remove(dist_file_path)

            torch.cuda.set_device(local_rank)
            init_process_group(
                backend="nccl",
                rank=process_id,
                world_size=world_size,
                init_method=f"file://{dist_file_path}",
            )
            print(f"[GPU {process_id}] READY")
        else:
            print("Running in single thread, DDP not enabled.")

    def setup_wandb(self):
        if self.is_main_node and self.config.wandb_project is not None:
            self.info.wandb_run_id = self.info.wandb_run_id or wandb.util.generate_id()
            wandb.init(project=self.config.wandb_project, entity=self.config.wandb_entity, name=self.config.experiment_id, id=self.info.wandb_run_id, resume="allow", config=self.config.to_dict())

            if self.info.total_steps > 0:
                wandb.alert(title=f"Training {self.info.wandb_run_id} resumed", text=f"Training {self.info.wandb_run_id} resumed from step {self.info.total_steps}")
            else:
                wandb.alert(title=f"Training {self.info.wandb_run_id} started", text=f"Training {self.info.wandb_run_id} started")

    # LOAD UTILITIES ----------
    def load_model(self, model, model_id=None, full_path=None, strict=True):
        if model_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{model_id}.{self.config.checkpoint_extension}"
        elif full_path is None and model_id is None:
            raise ValueError(
                "This method expects either 'model_id' or 'full_path' to be defined"
            )

        checkpoint = load_or_fail(full_path, wandb_run_id=self.info.wandb_run_id if self.is_main_node else None)
        if checkpoint is not None:
            model.load_state_dict(checkpoint, strict=strict)
            del checkpoint

        return model

    def load_optimizer(self, optim, optim_id=None, full_path=None, fsdp_model=None):
        if optim_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{optim_id}.pt"
        elif full_path is None and optim_id is None:
            raise ValueError(
                "This method expects either 'optim_id' or 'full_path' to be defined"
            )

        checkpoint = load_or_fail(full_path, wandb_run_id=self.info.wandb_run_id if self.is_main_node else None)
        if checkpoint is not None:
            try:
                if fsdp_model is not None:
                    sharded_optimizer_state_dict = (
                        FSDP.scatter_full_optim_state_dict(  # <---- FSDP
                            checkpoint
                            if (
                                self.is_main_node
                                or self.fsdp_defaults["sharding_strategy"]
                                == ShardingStrategy.NO_SHARD
                            )
                            else None,
                            fsdp_model,
                        )
                    )
                    optim.load_state_dict(sharded_optimizer_state_dict)
                    del checkpoint, sharded_optimizer_state_dict
                else:
                    optim.load_state_dict(checkpoint)
            # pylint: disable=broad-except
            except Exception as e:
                print("!!! Failed loading optimizer, skipping... Exception:", e)

        return optim

    # SAVE UTILITIES ----------
    def save_info(self, info, suffix=""):
        full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/info{suffix}.json"
        create_folder_if_necessary(full_path)
        if self.is_main_node:
            safe_save(vars(self.info), full_path)

    def save_model(self, model, model_id=None, full_path=None, is_fsdp=False):
        if model_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{model_id}.{self.config.checkpoint_extension}"
        elif full_path is None and model_id is None:
            raise ValueError(
                "This method expects either 'model_id' or 'full_path' to be defined"
            )
        create_folder_if_necessary(full_path)
        if is_fsdp:
            with FSDP.summon_full_params(model):
                pass
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, self.fsdp_fullstate_save_policy
            ):
                checkpoint = model.state_dict()
            if self.is_main_node:
                safe_save(checkpoint, full_path)
            del checkpoint
        else:
            if self.is_main_node:
                checkpoint = model.state_dict()
                safe_save(checkpoint, full_path)
                del checkpoint

    def save_optimizer(self, optim, optim_id=None, full_path=None, fsdp_model=None):
        if optim_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{optim_id}.pt"
        elif full_path is None and optim_id is None:
            raise ValueError(
                "This method expects either 'optim_id' or 'full_path' to be defined"
            )
        create_folder_if_necessary(full_path)
        if fsdp_model is not None:
            optim_statedict = FSDP.full_optim_state_dict(fsdp_model, optim)
            if self.is_main_node:
                safe_save(optim_statedict, full_path)
            del optim_statedict
        else:
            if self.is_main_node:
                checkpoint = optim.state_dict()
                safe_save(checkpoint, full_path)
                del checkpoint
    # -----

    def __init__(self, config_file_path=None, config_dict=None, device="cpu", training=True):
        # Temporary setup, will be overriden by setup_ddp if required
        self.device = device
        self.process_id = 0
        self.is_main_node = True
        self.world_size = 1
        # ----

        self.config: self.Config = self.setup_config(config_file_path, config_dict, training)
        self.info: self.Info = self.setup_info()

    def __call__(self, single_gpu=False):
        self.setup_ddp(self.config.experiment_id, single_gpu=single_gpu)  # this will change the device to the CUDA rank
        self.setup_wandb()
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.is_main_node:
            print()
            print("**STARTIG JOB WITH CONFIG:**")
            print(yaml.dump(self.config.to_dict(), default_flow_style=False))
            print("------------------------------------")
            print()
            print("**INFO:**")
            print(yaml.dump(vars(self.info), default_flow_style=False))
            print("------------------------------------")
            print()

        # SETUP STUFF
        extras = self.setup_extras_pre()
        assert extras is not None, "setup_extras_pre() must return a DTO"

        data = self.setup_data(extras)
        assert data is not None, "setup_data() must return a DTO"
        if self.is_main_node:
            print("**DATA:**")
            print(yaml.dump({k:type(v).__name__ for k, v in data.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        models = self.setup_models(extras)
        assert models is not None, "setup_models() must return a DTO"
        if self.is_main_node:
            print("**MODELS:**")
            print(yaml.dump({
                k:f"{type(v).__name__} - {f'trainable params {sum(p.numel() for p in v.parameters() if p.requires_grad)}' if isinstance(v, nn.Module) else 'Not a nn.Module'}" for k, v in models.to_dict().items()
            }, default_flow_style=False))
            print("------------------------------------")
            print()

        optimizers = self.setup_optimizers(extras, models)
        assert optimizers is not None, "setup_optimizers() must return a DTO"
        if self.is_main_node:
            print("**OPTIMIZERS:**")
            print(yaml.dump({k:type(v).__name__ for k, v in optimizers.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        schedulers = self.setup_schedulers(extras, models, optimizers)
        assert schedulers is not None, "setup_schedulers() must return a DTO"
        if self.is_main_node:
            print("**SCHEDULERS:**")
            print(yaml.dump({k:type(v).__name__ for k, v in schedulers.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        post_extras =self.setup_extras_post(extras, models, optimizers, schedulers)
        assert post_extras is not None, "setup_extras_post() must return a DTO"
        extras = self.Extras.from_dict({ **extras.to_dict(),**post_extras.to_dict() })
        if self.is_main_node:
            print("**EXTRAS:**")
            print(yaml.dump({k:f"{v}" for k, v in extras.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()
        # -------

        # TRAIN
        if self.is_main_node:
            print("**TRAINING STARTING...**")
        self.train(data, extras, models, optimizers, schedulers, single_gpu)

        if single_gpu is False:
            barrier()
            destroy_process_group()
        if self.is_main_node:
            print()
            print("------------------------------------")
            print()
            print("**TRAINING COMPLETE**")
            if self.config.wandb_project is not None:
                wandb.alert(title=f"Training {self.info.wandb_run_id} finished", text=f"Training {self.info.wandb_run_id} finished")
