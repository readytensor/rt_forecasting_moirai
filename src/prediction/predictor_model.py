import os
import warnings
import math
import joblib
import numpy as np
import pandas as pd
from typing import List, Union, Dict
from schema.data_schema import ForecastingSchema
from logger import get_logger
from uni2ts.model.moirai import MoiraiForecast

warnings.filterwarnings("ignore")

import torch
from prediction.download_model import download_pretrained_model_if_not_exists

logger = get_logger(task_name=__name__)
# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger.info(f"device used: {device}")

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"


class Forecaster:
    """Chronos Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    MODEL_NAME = "Chronos_Timeseries_Forecaster"
    SERIES_PER_FORECAST = 10

    def __init__(
        self,
        model_name: str,
        data_schema: ForecastingSchema,
        top_k: int = 50,
        top_p: float = 1,
        temperature: float = 0.0001,
        num_samples: int = 20,
        **kwargs,
    ):
        """Construct a new Chronos Forecaster."""
        self.model_name = model_name
        self.data_schema = data_schema
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = float(temperature)
        self.num_samples = num_samples
        self.kwargs = kwargs

    def fit(self) -> None:
        """Train the model."""
        pretrained_model_root_path = os.path.join(
            os.path.dirname(__file__), "pretrained_model", self.model_name
        )
        download_pretrained_model_if_not_exists(
            pretrained_model_root_path, model_name=self.model_name
        )
        return None

    def predict(self, context: List[torch.Tensor]) -> np.ndarray:
        """
        Generate forecast for future timesteps.
        Args:
        - context (List[torch.Tensor]): The context data.
        Returns (np.ndarray): The forecasted values.
        """
        # download model if not exists
        pretrained_model_root_path = os.path.join(
            os.path.dirname(__file__), "pretrained_model", self.model_name
        )

        self.model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=pretrained_model_root_path,
            map_location=device,
        )
        # we predict in batches to lower memory requirements
        num_batches = math.ceil(len(context) / self.SERIES_PER_FORECAST)
        all_predictions = []
        for i in range(num_batches):
            logger.info(f"Predicting for batch {i+1} out of {num_batches} batches")
            batch_context = context[
                i * self.SERIES_PER_FORECAST : (i + 1) * self.SERIES_PER_FORECAST
            ]
            batch_predictions = self.model.predict(
                context=batch_context,
                prediction_length=self.data_schema.forecast_length,
                num_samples=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                limit_prediction_length=False,
            )
            all_predictions.append(batch_predictions)

        all_predictions = np.concatenate(all_predictions)
        return np.array(all_predictions).mean(axis=1)

    def save(self, model_dir_path: str) -> None:
        """Save the model to the specified directory."""
        self.model = None
        os.makedirs(model_dir_path, exist_ok=True)
        model_file_path = os.path.join(model_dir_path, PREDICTOR_FILE_NAME)
        joblib.dump(self, model_file_path)

    @classmethod
    def load(self, model_file_path: str) -> "Forecaster":
        """Load the model from the specified directory."""
        model = joblib.load(model_file_path)
        pretrained_model_root_path = os.path.join(
            os.path.dirname(__file__), "pretrained_model", model.model_name
        )
        model.model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=pretrained_model_root_path,
            map_location=device,
        )
        return model

    def __str__(self):
        return f"Model name: {self.MODEL_NAME}"

    def preprocess_context(
        self, context: pd.DataFrame
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """Preprocess the context data."""
        series_id_col = self.data_schema.id_col
        target_col = self.data_schema.target

        grouped = context.groupby(series_id_col)

        all_ids = [i for i, _ in grouped]
        all_series = [i for _, i in grouped]

        if len(all_ids) == 1:
            processed_context = torch.tensor(
                all_series[0][target_col].to_numpy().reshape(1, -1)
            )

        else:
            processed_context = []
            for series in all_series:
                series = series[target_col].to_numpy().reshape(1, -1).flatten()
                series = torch.tensor(series)
                processed_context.append(series)

        return processed_context, all_ids


def predict_with_model(
    model: Forecaster,
    context: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Generate forecast.

    Args:
       - model (Forecaster): The predictor model.
       - context (pd.DataFrame): The context data.

    Returns:
       - Dict[str, np.ndarray]: Dictionary with the series id as keys and the forecast as values.
    """
    processed_context, ids = model.preprocess_context(context)
    predictions = model.predict(
        context=processed_context,
    )
    return {k: v for k, v in zip(ids, predictions)}


def train_predictor_model(model_name: str, **kwargs) -> Forecaster:
    """
    Train the predictor model.
    Args:
    - model_name (str): The name of the model to train.
    - **kwargs: Additional keyword arguments.

    Returns (Forecaster): The predictor model.
    """
    model = Forecaster(model_name=model_name, **kwargs)
    model.fit()
    return model


def save_predictor_model(model: Forecaster, predictor_file_path: str) -> None:
    """Save the predictor model to the specified path.
    Args:
    - model (Forecaster): The predictor model.
    - predictor_file_path (str): The path to save the model.
    Returns: None
    """
    model.save(predictor_file_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """Load the predictor model from the specified path.
    Args:
    - predictor_dir_path (str): The directory path to load the model from.
    Returns (Forecaster): The predictor model.
    """

    predictor_file_path = os.path.join(predictor_dir_path, PREDICTOR_FILE_NAME)
    return Forecaster.load(predictor_file_path)
