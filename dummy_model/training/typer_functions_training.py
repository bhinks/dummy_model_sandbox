import typer
import pandas as pd
from hydra import compose, initialize

from dummy_model.modules import DUMMY_IMPORT__MODULES


CONFIG_FILE_PATH = './'
with initialize(config_path=CONFIG_FILE_PATH, version_base=None):
    cfg = compose(config_name="config")

app = typer.Typer(chain=True)


@app.command('test_command')
def test_command() -> None:
    print(f'test cfg read: {cfg["paths"]["raw_data_path"]}')
    print(f'test modules import: {DUMMY_IMPORT__MODULES}')


@app.command("generate_dataset")
def generate_dataset() -> None:
    raise NotImplementedError


@app.command("log_dataset")
def log_dataset() -> None:
    raise NotImplementedError


@app.command("train_model")
def train_model() -> None:
    raise NotImplementedError


@app.command("log_model_artifacts")
def log_model_artifacts() -> None:
    raise NotImplementedError


@app.command("evaluate_model")
def evaluate_model() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    app()
