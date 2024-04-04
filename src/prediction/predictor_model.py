import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional, Union, List
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
from gluonts.dataset.common import ListDataset


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
    made_up_frequency = "S"  # by seconds
    made_up_start_dt = "2000-01-01 00:00:00"

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
        use_exogenous: Optional[bool] = True,
    ) -> None:
        super().__init__(prediction_length, lead_time=lead_time)
        self.data_schema = data_schema
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.context_length = context_length
        self.use_exogenous = use_exogenous
        self.freq = self.map_frequency(data_schema.frequency)
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
            return "D"

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

    def prepare_training_data(
        self,
        history: pd.DataFrame,
    ) -> List:
        """
        Applys the history_forecast_ratio parameter and puts the training data into the shape expected by GluonTS.

        Args:
            history (pd.DataFrame): The input dataset.

        Returns (ListDataset): The processed dataset expected by GluonTS.
        """
        data_schema = self.data_schema
        # Make sure there is a date column
        history = self.prepare_time_column(data=history, is_train=True)

        # Manage each series in the training data separately
        all_covariates = []
        groups_by_ids = history.groupby(data_schema.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=data_schema.id_col)
            for id_ in all_ids
        ]

        fut_cov_names = []
        past_cov_names = []
        static_cov_names = []

        if self.use_exogenous:
            fut_cov_names = data_schema.future_covariates
            past_cov_names = data_schema.past_covariates
            static_cov_names = data_schema.static_covariates

        # Put future covariates into separate list
        all_covariates = []
        static_covariates = []

        for series in all_series:
            series_past_covariates = []
            series_future_covariates = []
            series_static_covariates = []

            for covariate in fut_cov_names:
                series_future_covariates.append(series[covariate])

            for covariate in past_cov_names:
                series_past_covariates.append(series[covariate])

            for covariate in static_cov_names:
                series_static_covariates.append(series[covariate].iloc[0])

            all_covariates.append((series_future_covariates, series_past_covariates))
            static_covariates.append(series_static_covariates)

        # If covariates are available for training, create a dataset with covariate features,
        # otherwise a dataset with only target series will be created.

        start = (
            series[data_schema.time_col].iloc[0]
            if data_schema.time_col_dtype not in ["INT", "OTHER"]
            else "2020-01-01"
        )
        list_dataset = [
            {
                # "start": series[data_schema.time_col].iloc[0],
                "start": pd.Period(
                    start,
                    self.map_frequency(data_schema.frequency),
                ),
                "target": series[data_schema.target],
            }
            for series in all_series
        ]

        if self.use_exogenous and fut_cov_names:
            for item, cov_series in zip(list_dataset, all_covariates):
                item["feat_dynamic_real"] = cov_series[0]

        if self.use_exogenous and past_cov_names:
            for item, cov_series in zip(list_dataset, all_covariates):
                item["past_feat_dynamic_real"] = cov_series[1]

        if self.use_exogenous and static_cov_names:
            for item, cov_series in zip(list_dataset, static_covariates):
                item["feat_static_real"] = cov_series

        return list_dataset

    def prepare_time_column(
        self, data: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        """
        Adds time column of type DATETIME to datasets that have time column dtype as INT.

        Args:
            data (pd.DataFrame): The input dataset.
            is_train (bool): Set to true for training dataset and false for testing dataset.

            Returns (pd.DataFrame): The dataset after processing time column.
        """
        # sort data
        time_col_dtype = self.data_schema.time_col_dtype
        id_col = self.data_schema.id_col
        time_col = self.data_schema.time_col

        data = data.sort_values(by=[id_col, time_col])

        if time_col_dtype == "INT":
            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            if is_train:
                # since GluonTS requires a date column, we will make up a timeline
                start_date = pd.Timestamp(self.made_up_start_dt)
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
                self.last_timestamp = datetimes[-1]
                self.timedelta = datetimes[-1] - datetimes[-2]

            else:
                start_date = self.last_timestamp + self.timedelta
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
            int_vals = sorted(data[time_col].unique().tolist())
            self.time_to_int_map = dict(zip(datetimes, int_vals))
            # Repeat the datetime range for each location
            data[time_col] = list(datetimes) * num_series
        else:
            data[time_col] = pd.to_datetime(data[time_col])
            data[time_col] = data[time_col].dt.tz_localize(None)

        return data


def train_predictor_model() -> None:
    return None


def predict_with_model(model: MoiraiPredictor, context: pd.DataFrame):
    schema = model.data_schema
    grouped = context.groupby(schema.id_col)
    all_ids = [id for id, _ in grouped]
    all_series = [series for _, series in grouped]

    all_forecasts = []
    data = model.prepare_training_data(history=context)
    for i in data:
        print(ListDataset([i], model.freq))
        forecast = list(
            model.predict(
                ListDataset([i], model.freq),
                context_length=model.context_length,
            )
        )[0].samples

        median_forecast = np.median(forecast, axis=0)

        all_forecasts.append(median_forecast)
    all_forecasts = np.array(all_forecasts)
    print("all_forecasts:", all_forecasts.shape)
    return {k: v for k, v in zip(all_ids, all_forecasts)}


def save_predictor_model(model: MoiraiPredictor, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
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
