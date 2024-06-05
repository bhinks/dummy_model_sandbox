# ML-Cookie-Cutter-Template

A minimal template for the development and deployment of data science models.

## Dependency Management
Before you start working on your project: 
1. Run `make setup_dev_env` to create your python virtual environment
2. Then run `source $(poetry env info --path)/bin/activate` to activate it 

These steps will generate a `poetry.lock` file; keep it in the repo, as it will speed up the re-building of the environment if you choose to do so, as well as during deployment.

> **TECHNICAL NOTE** - Python dependencies are maintained using [Poetry](https://python-poetry.org/). Make sure to track your python dependencies in the `pyproject.toml` file.

> **TECHNICAL NOTE** - This template assumes that the python versions on your system are managed by `pyenv`; the `local` python version `pyenv` should use for this folder is specified in the `.python-version` file (the same python version is configured in the `pyproject.toml`'s `tool.poetry.dependencies` field). If you're working in SageMaker Studio on Datalab, you don't have to worry about this note as `pyenv` is already installed and configured there by default.

## Batch Inference Handoff
To ensure smooth deployment, you need to:
1. Make sure your python dependencies are tracked in the `pyproject.toml` file under `[tool.poetry.dependencies]`.
    - If you were tracking your dependencies using `conda`:
        - `conda activate <env>`
        - `conda install pip`
        - `pip list --format=freeze > requirements.txt`
        - `poetry add $( cat requirements.txt )`
    - If you were tracking your dependencies using `pip`:
        - `poetry add $( cat requirements.txt )`
    - If you were tracking you dependecies using `poetry`:
        - `poetry show --why > some_file.txt`
        - Then copy the contents of `some_file.txt` into Notepad++ and Find `^([a-z0-9-]+)[\s\(\)\!]*([0-9\.]+).*$` and Replace `$1 = "$2"` with Regular Expression enabled.
        - Copy the final result under the `[tool.poetry.dependencies]` heading in the `pyproject.toml`; get rid of the `[tool.poetry.extras]` heading.
    - *References*:
        - https://stackoverflow.com/a/57845418
        - https://stackoverflow.com/a/67335464
        - https://stackoverflow.com/a/64787419
2. Fill in your code/logic wherever you see `### REPLACE THIS ###` in `dummy_model/batch_inference/typer_functions_inference.py`. This file defines the actual functionality to run during the batch inference pipeline.
3. For setting up monitoring (after batch inference deployment is complete), follow the steps [here](https://github.com/DTS-GDA-BI-Platform/GENERATOR-batch-inference-processing-containers/tree/main/%7B%7B%20cookiecutter.project_name%20%7D%7D/%7B%7B%20cookiecutter.project_slug%20%7D%7D-monitoring/code).

A few things to keep in mind as you work on these steps:
- The configs defined in `dummy_model/batch_inference/config.yaml` are accessible in the typer functions files; this is achieved using [`hydra`](https://hydra.cc/docs/intro/).
- Python functions/files that your code uses should be put in the `modules` folders. As an exmaple, a dummy value `DUMMY_VALUE__MODULES` is imported from `dummy_model/modules/__init__.py` into the `dummy_model/batch_inference/typer_functions_inference.py` file.

# Dummy Model
*Add documentation specific to your model here.*
