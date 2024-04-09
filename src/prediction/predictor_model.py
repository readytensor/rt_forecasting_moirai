import os
import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
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
from uni2ts.data.builder.simple import (
    SimpleDatasetBuilder,
    SimpleEvalDatasetBuilder,
    generate_eval_builders,
)
from uni2ts.data.builder import ConcatDatasetBuilder
from uni2ts.model.moirai.finetune import (
    TrainDataLoader,
    ValidationDataLoader,
    # MoiraiFinetuneSmall,
    FinetuneTrainer,
    MoiraiFinetune,
)
from config import paths
from schema.data_schema import load_json_data_schema

count_patches = {}


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
        model_name: str,
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
        self.model_name = model_name
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

    def prepare_data(self) -> None:
        file_name = [i for i in os.listdir(paths.TRAIN_DIR) if i.endswith(".csv")][0]
        file_path = os.path.join(paths.TRAIN_DIR, file_name)
        dataset = file_name.removesuffix(".csv")
        freq = self.map_frequency(self.data_schema.frequency)
        offset = None
        SimpleDatasetBuilder(dataset=dataset).build_dataset(
            file=Path(file_path),
            offset=offset,
            date_offset=None,
            id_col=self.data_schema.id_col,
            time_col=self.data_schema.time_col,
            target_col=self.data_schema.target,
            freq=freq,
        )
        if offset is not None:

            SimpleEvalDatasetBuilder(
                f"{dataset}_eval",
                offset=offset,
                windows=None,
                distance=None,
                prediction_length=self.data_schema.forecast_length,
                context_length=None,
                patch_size=None,
            ).build_dataset(
                file=Path(file_path),
                id_col=self.data_schema.id_col,
                time_col=self.data_schema.time_col,
                target_col=self.data_schema.target,
                freq=freq,
            )
        self.dataset = dataset

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

            count_patches[patch_size] = count_patches.get(patch_size, 0) + 1

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

    def fit(self) -> MoiraiFinetune:
        model = MoiraiFinetune.get_model(self.model_name)

        trainer = FinetuneTrainer()

        dataset = SimpleDatasetBuilder(dataset=self.dataset).load_dataset(
            model.create_train_transform()
        )

        # val_dataset = SimpleDatasetBuilder(dataset=f"{self.dataset}_eval").load_dataset(
        #     model.create_val_transform
        # )
        # val_dataset = ConcatDatasetBuilder(
        #     *generate_eval_builders(
        #         dataset="banks_split_eval",
        #         offset=2000,
        #         eval_length=200,
        #         prediction_lengths=[96, 192, 336, 720],
        #         context_lengths=[1000, 2000, 3000, 4000, 5000],
        #         patch_sizes=[32, 64],
        #     )
        # ).load_dataset(model.create_val_transform)
        train_dataloader = TrainDataLoader(dataset=dataset, trainer=trainer)
        # val_dataloader = ValidationDataLoader(dataset=val_dataset, trainer=trainer)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=None)
        self.model = model
        return model

    def save(self, save_dir_path: str) -> None:
        del self.prediction_net
        self.serialize
        joblib.dump(self, os.path.join(save_dir_path, "predictor.joblib"))

    def serialize(self, path: Path) -> None:
        super().serialize(path)

        # Save ckpt
        with open(path / "model.ckpt", "wb") as fp:
            torch.save(
                {
                    "state_dict": self.prediction_net.state_dict(),
                    "hyper_parameters": {
                        "module_kwargs": self.prediction_net.hparams.module_kwargs
                    },
                },
                fp,
            )

        # Save Predictor params
        with open(path / "predictor_config.json", "w") as fp:
            json.dump(
                {
                    "batch_size": self.batch_size,
                    "prediction_length": self.prediction_length,
                    "context_length": self.context_length,
                    "patch_size": self.patch_size,
                    "num_samples": self.num_samples,
                    "lead_time": self.lead_time,
                },
                fp,
                indent=4,
            )

        joblib.dump(self.data_schema, path / "data_schema.joblib")

    @classmethod
    def deserialize(
        cls,
        path: Path,
        # prediction_length=64,
        # context_length=1024,
        # patch_size="auto",
        # num_parallel_samples=20,
        # batch_size=32,
    ) -> "MoiraiPredictor":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(path / "predictor_config.json", "r") as fp:
            predictor_config: dict = json.load(fp)

        # predictor_config.update(
        #     prediction_length=prediction_length,
        #     batch_size=batch_size,
        # )

        with open(path / "model.ckpt", "rb") as fp:
            ckpt = torch.load(fp, map_location=device)

        batch_size = 16
        if "batch_size" in predictor_config.keys():
            batch_size = predictor_config.pop("batch_size")

        lead_time = 0
        if "lead_time" in predictor_config.keys():
            lead_time = predictor_config.pop("lead_time")

        model = CustomizableMoiraiForecast(
            module_kwargs=ckpt["hyper_parameters"]["module_kwargs"],
            # context_length=context_length,
            # patch_size=patch_size,
            # num_samples=num_parallel_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            **predictor_config,
        )
        model.load_state_dict(ckpt["state_dict"])
        data_schema = joblib.load(path / "data_schema.joblib")

        return MoiraiPredictor(
            model_name="loaded_model",
            prediction_net=model,
            data_schema=data_schema,
            batch_size=batch_size,
            lead_time=lead_time,
            **predictor_config,
        )


def train_predictor_model(
    model_name: str, data_schema: ForecastingSchema, **kwargs
) -> MoiraiPredictor:

    model = load_pretrained_model(
        model_name=model_name,
        data_schema=data_schema,
        **kwargs,
    )
    model.prepare_data()
    model.fit()

    return model


def predict_with_model(model: MoiraiPredictor, context: pd.DataFrame):
    schema = model.data_schema
    grouped = context.groupby(schema.id_col)
    all_ids = [id for id, _ in grouped]
    all_series = [series for _, series in grouped]

    all_forecasts = []
    for _, series in zip(all_ids, all_series):
        start = (
            series[schema.time_col].iloc[0]
            if schema.time_col_dtype not in ["INT", "OTHER"]
            else "2020-01-01"
        )
        forecast = list(
            model.predict(
                dataset=[
                    {
                        "target": series[schema.target],
                        "start": pd.Period(
                            start,
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
    model.serialize(Path(model_dir))


def load_pretrained_model(
    model_name: str, data_schema: ForecastingSchema, **kwargs
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
            prediction_length=data_schema.forecast_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
            **kwargs,
        ),
        data_schema=data_schema,
        prediction_length=data_schema.forecast_length,
        model_name=model_name,
        **kwargs,
    )

    return model


def load_predictor_model(save_dir_path: str):
    return MoiraiPredictor.deserialize(Path(save_dir_path))
