from pathlib import Path

def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "N": 6,
        "h": 8,
        "dropout": 0.1,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_target": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config["datasource"]}_{config["model_folder"]}"
    model_filename = f"{config["model_basename"]}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

def get_latest_weights_file_path(config):
    model_folder = f"{config["datasource"]}_{config["model_folder"]}"
    model_filename = f"{config["model_basename"]}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
