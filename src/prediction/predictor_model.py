import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import SampleForecastGenerator
from gluonts.model.predictor import Predictor
from gluonts.torch.batchify import batchify
from gluonts.transform import SelectFields, TestSplitSampler
from gluonts.transform.split import TFTInstanceSplitter
from uni2ts.model.moirai.forecast import MoiraiForecast
from schema.data_schema import ForecastingSchema
from prediction.download_model import download_pretrained_model_if_not_exists


class CustomizableMoiraiForecast(MoiraiForecast):
    @contextmanager
    def custom_config(
        self,
        context_length: Optional[int] = None,
        patch_size: Optional[int] = None,
        num_parallel_samples: Optional[int] = None,
        prediction_length: Optional[int] = None,
    ):
        old_hparams = deepcopy(self.hparams)
        if context_length is not None:
            self.hparams["context_length"] = context_length
        if patch_size is not None:
            self.hparams["patch_size"] = patch_size
        if num_parallel_samples is not None:
            self.hparams["num_samples"] = num_parallel_samples
        if prediction_length is not None:
            self.hparams["prediction_length"] = prediction_length
        yield self

        self.hparams["context_length"] = old_hparams["context_length"]
        self.hparams["patch_size"] = old_hparams["patch_size"]
        self.hparams["num_samples"] = old_hparams["num_samples"]
        self.hparams["prediction_length"] = old_hparams["prediction_length"]


class MoiraiPredictor(Predictor):
    def __init__(
        self,
        data_schema: ForecastingSchema,
        prediction_net: CustomizableMoiraiForecast,
        prediction_length: int,
        lead_time: int = 0,
        patch_size: Optional[Union[int, str]] = None,
        num_samples: Optional[int] = 20,
        batch_size: Optional[int] = 16,
        context_length: Optional[int] = 50,
    ) -> None:
        super().__init__(prediction_length, lead_time=lead_time)
        self.data_schema = data_schema
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.context_length = context_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forecast_generator = SampleForecastGenerator()
        self.prediction_net = prediction_net.to(self.device)
        self.required_fields = ["forecast_start", "item_id", "info"]
        assert prediction_length == self.prediction_net.hparams.prediction_length

    @property
    def network(self) -> nn.Module:
        return self.prediction_net

    def _get_patch_size_val_loss_map(self, loader, prediction_net):
        patch_val_loss_map = {}
        for patch_size in prediction_net.module.patch_sizes:
            val_losses = []
            for batch in loader:
                past_target = batch["past_target"].to(self.device)
                past_observed_target = batch["past_observed_target"].to(self.device)
                past_is_pad = batch["past_is_pad"].to(self.device)
                val_losses.append(
                    prediction_net._val_loss(
                        patch_size=patch_size,
                        target=past_target[..., : prediction_net.past_length, :],
                        observed_target=past_observed_target[
                            ..., : prediction_net.past_length, :
                        ],
                        is_pad=past_is_pad[..., : prediction_net.past_length],
                        feat_dynamic_real=None,
                        observed_feat_dynamic_real=None,
                        past_feat_dynamic_real=None,
                        past_observed_feat_dynamic_real=None,
                    )
                )
            val_losses = torch.cat(val_losses, dim=0)
            patch_val_loss_map[patch_size] = val_losses.mean(0).cpu().numpy().item()
        return patch_val_loss_map

    def _get_best_patch_size(
        self,
        dataset,
        batch_size,
        context_length,
        prediction_length,
    ):

        with self.prediction_net.custom_config(
            context_length=context_length,
            patch_size="auto",
            prediction_length=prediction_length,
        ) as prediction_net:
            instance_splitter = TFTInstanceSplitter(
                instance_sampler=TestSplitSampler(),
                past_length=prediction_net.past_length,
                future_length=prediction_length,
                observed_value_field="observed_target",
                time_series_fields=[],
                past_time_series_fields=[],
            )
            input_transform = prediction_net.get_default_transform() + instance_splitter
            inference_data_loader = InferenceDataLoader(
                dataset,
                transform=input_transform
                + SelectFields(
                    prediction_net.prediction_input_names + self.required_fields,
                    allow_missing=True,
                ),
                batch_size=batch_size,
                stack_fn=lambda data: batchify(data, self.device),
            )

            prediction_net.eval()
            with torch.no_grad():
                patch_val_loss_map = self._get_patch_size_val_loss_map(
                    inference_data_loader, prediction_net=prediction_net
                )
                print("patch_val_loss_map:", patch_val_loss_map)
                return min(patch_val_loss_map, key=patch_val_loss_map.get)

    def predict(
        self,
        dataset: Dataset,
        context_length: Optional[int] = None,
        num_parallel_samples: Optional[int] = None,
    ) -> Iterator[Forecast]:

        prediction_length = self.data_schema.forecast_length
        patch_size = (
            self.patch_size
            if self.patch_size is not None
            else self.prediction_net.hparams.patch_size
        )
        if self.patch_size == "auto_dataset":
            patch_size = self._get_best_patch_size(
                dataset, self.batch_size, context_length, prediction_length
            )
            print("Selected patch_size:", patch_size)

        with self.prediction_net.custom_config(
            context_length=context_length,
            patch_size=patch_size,
            num_parallel_samples=num_parallel_samples,
            prediction_length=prediction_length,
        ) as prediction_net:
            instance_splitter = TFTInstanceSplitter(
                instance_sampler=TestSplitSampler(),
                past_length=prediction_net.past_length,
                future_length=prediction_length,
                observed_value_field="observed_target",
                time_series_fields=[],
                past_time_series_fields=[],
            )
            input_transform = prediction_net.get_default_transform() + instance_splitter
            inference_data_loader = InferenceDataLoader(
                dataset,
                transform=input_transform
                + SelectFields(
                    prediction_net.prediction_input_names + self.required_fields,
                    allow_missing=True,
                ),
                batch_size=self.batch_size,
                stack_fn=lambda data: batchify(data, self.device),
            )

            prediction_net.eval()
            with torch.no_grad():
                yield from self.forecast_generator(
                    inference_data_loader=inference_data_loader,
                    prediction_net=prediction_net,
                    input_names=prediction_net.prediction_input_names,
                    output_transform=None,
                    num_samples=self.num_samples,
                )

    def map_frequency(self, frequency: str) -> str:

        if self.data_schema.time_col_dtype == "INT":
            return 1

        frequency = frequency.lower()
        if frequency == "yearly":
            return "Y"
        if frequency == "quarterly":
            return "Q"
        if frequency == "monthly":
            return "M"
        if frequency == "weekly":
            return "W"
        if frequency == "daily":
            return "D"
        if frequency == "hourly":
            return "H"
        if frequency == "minutely":
            return "min"
        if frequency == "secondly":
            return "S"
        else:
            return 1


def train_predictor_model() -> None:

    return None


def predict_with_model(model: MoiraiPredictor, context: pd.DataFrame):
    schema = model.data_schema
    grouped = context.groupby(schema.id_col)
    all_ids = [id for id, _ in grouped]
    all_series = [series for _, series in grouped]

    all_forecasts = []
    for _, series in zip(all_ids, all_series):
        forecast = list(
            model.predict(
                dataset=[
                    {
                        "target": series[schema.target],
                        "start": pd.Period(
                            series[schema.time_col].iloc[0],
                            model.map_frequency(schema.frequency),
                        ),
                    }
                ],
                context_length=model.context_length,
            )
        )[0].samples

        median_forecast = np.median(forecast, axis=0)

        all_forecasts.append(median_forecast)
    all_forecasts = np.array(all_forecasts)
    return {k: v for k, v in zip(all_ids, all_forecasts)}


def save_predictor_model(model: MoiraiPredictor, model_dir: str) -> None:
    model_path = os.path.join(model_dir, "model.txt")
    with open(model_path, "w") as f:
        f.write("dummy model")


def load_predictor_model(
    model_name: str, data_schema: ForecastingSchema, prediction_length: int, **kwargs
) -> MoiraiPredictor:

    pretrained_model_root_path = os.path.join(
        os.path.dirname(__file__), "pretrained_model", model_name
    )

    download_pretrained_model_if_not_exists(
        pretrained_model_root_path, model_name=model_name
    )

    ckpt_path = Path(os.path.join(pretrained_model_root_path, "model.ckpt"))

    model = MoiraiPredictor(
        prediction_net=CustomizableMoiraiForecast.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
            **kwargs,
        ),
        data_schema=data_schema,
        prediction_length=prediction_length,
        **kwargs,
    )

    return model
