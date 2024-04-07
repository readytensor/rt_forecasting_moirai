#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import nn

from uni2ts.loss.packed import PackedDistributionLoss, PackedNLLLoss
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    EvalCrop,
    EvalMaskedPrediction,
    EvalPad,
    ExtendMask,
    FixedPatchSizeConstraints,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

from .module import MoiraiModule
from huggingface_hub import hf_hub_download
from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)
from uni2ts.data.loader import DataLoader, PackCollate
from torch.utils.data import Dataset, DistributedSampler

from config import paths


class MoiraiFinetune(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        module_kwargs: dict[str, Any],
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
    ):
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters()
        self.module = MoiraiModule(**module_kwargs)

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "*batch sample seq_len max_patch"]:
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        )
        preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
        return rearrange(preds, "n b ... -> b n ...")

    def loss(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, ""]:
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            patch_size,
        )
        loss = self.hparams.loss_func(
            pred=distr,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id,
        )
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.loss(**batch)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            self.hparams.loss_func.__class__.__name__,
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.loss(**batch)
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            "val_loss",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def configure_optimizers(self) -> dict:
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    def create_train_transform(self) -> Transformation:
        return (
            SampleDimension(
                max_dim=self.hparams.max_dim,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + GetPatchSize(
                min_time_patches=self.hparams.min_patches,
                target_field="target",
                patch_sizes=self.module.patch_sizes,
                patch_size_constraints=DefaultPatchSizeConstraints(),
                offset=True,
            )
            + PatchCrop(
                min_time_patches=self.hparams.min_patches,
                max_patches=self.module.max_seq_len,
                will_flatten=True,
                offset=True,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            + Patchify(
                max_patch_size=max(self.module.patch_sizes),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=self.hparams.max_dim,
                randomize=True,
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + MaskedPrediction(
                min_mask_ratio=self.hparams.min_mask_ratio,
                max_mask_ratio=self.hparams.max_mask_ratio,
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(self.seq_fields))
        )

    def create_val_transform(
        self,
        offset: int,
        distance: int,
        prediction_length: int,
        context_length: int,
        patch_size: int,
    ) -> Transformation:
        return (
            GetPatchSize(
                min_time_patches=2,
                target_field="target",
                patch_sizes=self.module.patch_sizes,
                patch_size_constraints=FixedPatchSizeConstraints(patch_size),
                offset=True,
            )
            + EvalCrop(
                offset,
                distance,
                prediction_length,
                context_length,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + PackFields(
                output_field="target",
                fields=("target",),
            )
            + PackFields(
                output_field="past_feat_dynamic_real",
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
            )
            + EvalPad(
                prediction_pad=-prediction_length % patch_size,
                context_pad=-context_length % patch_size,
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddObservedMask(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                observed_mask_field="observed_mask",
                collection_type=dict,
            )
            + ImputeTimeSeries(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                imputation_method=DummyValueImputation(value=0.0),
            )
            + Patchify(
                max_patch_size=max(self.module.patch_sizes),
                fields=("target", "observed_mask"),
                optional_fields=("past_feat_dynamic_real",),
            )
            + AddVariateIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                variate_id_field="variate_id",
                expected_ndim=3,
                max_dim=self.hparams.max_dim,
                randomize=True,
                collection_type=dict,
            )
            + AddTimeIndex(
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                time_id_field="time_id",
                expected_ndim=3,
                collection_type=dict,
            )
            + EvalMaskedPrediction(
                mask_length=-prediction_length % patch_size,
                target_field="target",
                truncate_fields=("variate_id", "time_id", "observed_mask"),
                optional_truncate_fields=("past_feat_dynamic_real",),
                prediction_mask_field="prediction_mask",
                expected_ndim=3,
            )
            + ExtendMask(
                fields=tuple(),
                optional_fields=("past_feat_dynamic_real",),
                mask_field="prediction_mask",
                expected_ndim=3,
            )
            + FlatPackCollection(
                field="variate_id",
                feat=False,
            )
            + FlatPackCollection(
                field="time_id",
                feat=False,
            )
            + FlatPackCollection(
                field="prediction_mask",
                feat=False,
            )
            + FlatPackCollection(
                field="observed_mask",
                feat=True,
            )
            + FlatPackFields(
                output_field="target",
                fields=("target",),
                optional_fields=("past_feat_dynamic_real",),
                feat=True,
            )
            + SequencifyField(field="patch_size", target_field="target")
            + SelectFields(fields=list(self.seq_fields))
        )


class MoiraiLinearProbe(MoiraiFinetune): ...


class FinetuneTrainer(L.Trainer):
    save_dir = paths.OUTPUT_DIR

    def __init__(self):
        super().__init__(
            accelerator="auto",
            strategy="auto",
            devices="auto",
            num_nodes=1,
            precision=32,
            logger=TensorBoardLogger(save_dir=self.save_dir, name="logs"),
            callbacks=[
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    dirpath=f"{self.save_dir}/checkpoints",
                    monitor="val_loss",
                    save_weights_only=True,
                    mode="min",
                    save_top_k=1,
                    every_n_epochs=1,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.0,
                    patience=3,
                    mode="min",
                    strict=False,
                ),
            ],
            max_epochs=100,
            enable_progress_bar=True,
            accumulate_grad_batches=1,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )


class MoiraiFinetuneSmall(MoiraiFinetune):
    def __init__(
        self,
    ):

        module_kwargs = {
            "distr_output": MixtureOutput(
                components=[
                    StudentTOutput(),
                    NormalFixedScaleOutput(),
                    NegativeBinomialOutput(),
                    LogNormalOutput(),
                ]
            ),
            "d_model": 384,
            "num_layers": 6,
            "patch_sizes": tuple([8, 16, 32, 64, 128]),
            "max_seq_len": 512,
            "attn_dropout_p": 0.0,
            "dropout_p": 0.0,
            "scaling": True,
        }
        args = {
            "min_patches": 2,
            "min_mask_ratio": 0.15,
            "max_mask_ratio": 0.5,
            "max_dim": 128,
            "loss_func": PackedNLLLoss(),
            "lr": 1e-3,
            "weight_decay": 1e-1,
            "beta1": 0.9,
            "beta2": 0.98,
            # "num_training_steps": 10000,
            "num_training_steps": 1,
            "num_warmup_steps": 0,
            # "checkpoint_path": hf_hub_download(
            #     repo_id="Salesforce/moirai-1.0-R-small", filename="model.ckpt"
            # ),
        }
        super().__init__(module_kwargs=module_kwargs, **args)

    @staticmethod
    def get_model():
        return MoiraiFinetuneSmall.load_from_checkpoint(
            checkpoint_path=hf_hub_download(
                repo_id="Salesforce/moirai-1.0-R-small", filename="model.ckpt"
            ),
        )


class ValidationDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, trainer: L.LightningModule):
        self.dataset = dataset
        sampler = None
        if trainer.world_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=None,
                rank=None,
                shuffle=True,
                seed=0,  # Assuming your DistributedSampler supports a 'seed' argument
                drop_last=False,
            )

        super().__init__(
            dataset=dataset,
            batch_size=32,
            batch_size_factor=2.0,
            num_batches_per_epoch=1,
            cycle=True,
            shuffle=True,
            num_workers=11,
            collate_fn=PackCollate(
                max_length=512,
                pad_func_map=MoiraiFinetune.pad_func_map,
                seq_fields=MoiraiFinetune.seq_fields,
            ),
            pin_memory=True,
            drop_last=False,
            fill_last=False,
            worker_init_fn=None,
            prefetch_factor=2,
            persistent_workers=True,
            sampler=sampler,
        )


class TrainDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, trainer: L.LightningModule):
        self.dataset = dataset
        sampler = None
        if trainer.world_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=trainer.world_size,
                rank=trainer.global_rank,
                shuffle=True,
                seed=0,  # Assuming your DistributedSampler supports a 'seed' argument
                drop_last=False,
            )

        super().__init__(
            dataset=dataset,
            batch_size=32,
            batch_size_factor=2.0,
            num_batches_per_epoch=100,
            cycle=True,
            shuffle=True,
            num_workers=11,
            collate_fn=PackCollate(
                max_length=512,
                pad_func_map=MoiraiFinetune.pad_func_map,
                seq_fields=MoiraiFinetune.seq_fields,
            ),
            pin_memory=True,
            drop_last=False,
            fill_last=False,
            worker_init_fn=None,
            prefetch_factor=2,
            persistent_workers=True,
            sampler=sampler,
        )
