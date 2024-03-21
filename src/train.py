from config import paths
from logger import get_logger, log_error
from schema.data_schema import load_json_data_schema
from prediction.predictor_model import train_predictor_model, save_predictor_model
from utils import ResourceTracker, set_seeds, read_json_as_dict


logger = get_logger(task_name="train")


def run_training(
    input_schema_dir_path: str = paths.INPUT_SCHEMA_DIR,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
) -> None:
    """
    Train the predictor model and saves it in the specified path.
    Args:
        - input_schema_dir_path (str): The path to the input schema directory.
        - model_config_file_path (str): The path to the model config file.
        - predictor_dir_path (str): The path to the predictor directory.
        - default_hyperparameters_file_path (str): The path to the default hyperparameters file.
    Returns:
        - None
    """

    try:
        with ResourceTracker(logger=logger, monitoring_interval=0.0001):
            logger.info("Starting training...")
            data_schema = load_json_data_schema(input_schema_dir_path)

            # load model config
            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)

            # set seeds
            logger.info("Setting seeds...")
            set_seeds(seed_value=model_config["seed_value"])

            # use default hyperparameters to train model
            logger.info("Loading hyperparameters...")
            default_hyperparameters = read_json_as_dict(
                default_hyperparameters_file_path
            )

            # use default hyperparameters to train model
            logger.info(f"Training model ({model_config['model_name']})...")
            model = train_predictor_model(
                model_name=model_config["model_name"],
                data_schema=data_schema,
                **default_hyperparameters,
            )

        # save predictor model
        logger.info("Saving model...")
        save_predictor_model(model, predictor_dir_path)

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
