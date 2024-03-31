import os
import requests
import json


def download_pretrained_model_if_not_exists(directory_path, model_name):
    print(f"Downloading pretrained model {model_name}...")
    files_urls = {
        "model.ckpt": f"https://huggingface.co/Salesforce/{model_name}/resolve/main/model.ckpt",
    }

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for file_name, url in files_urls.items():
        file_path = os.path.join(directory_path, file_name)
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {file_name}...")
                response = requests.get(url, allow_redirects=True)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except requests.RequestException as e:
                raise ValueError(
                    f"Error downloading pretrained model file from {file_path}."
                ) from e
        else:
            print(f"-- {file_name} already exists.")

    print(f"Model `{model_name}` is ready.")


if __name__ == "__main__":
    # relative path to config file
    config_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "model_config.json"
    )

    with open(config_file_path) as f:
        model_config = json.load(f)
        model_name = model_config["model_name"]

    pretrained_model_root_path = os.path.join(
        os.path.dirname(__file__), "pretrained_model", model_name
    )
    download_pretrained_model_if_not_exists(pretrained_model_root_path, model_name)
