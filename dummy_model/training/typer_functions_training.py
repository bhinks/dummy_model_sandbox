import os
from functools import wraps

import typer
import mlflow
import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from hydra import compose, initialize

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ds_snowflake_utils import SnowflakeInterface

from dummy_model.modules import DUMMY_IMPORT__MODULES
from dummy_model.modules.preprocessing import preprocess_data


with initialize(config_path='./config/', version_base=None):
    cfg = compose(config_name=f'{os.environ.get("DATALAB_DEPLOYMENT_ENVIRONMENT", "LOCAL")}')


snowflake_interface = SnowflakeInterface(
    model='dummy_model',
    environment=os.environ.get('DATALAB_DEPLOYMENT_ENVIRONMENT', 'LOCAL')
)

app = typer.Typer(chain=True)


@app.command('test_command')
def test_command() -> None:
    print(f'Test cfg read: {cfg.environment.data_paths.raw_data_path}')
    print(f'Test modules import: {DUMMY_IMPORT__MODULES}')


mlflow.set_tracking_uri(cfg.environment.mlflow.tracking_server_url)
def mlflow_run(wrapped_function):
    @wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        mlflow.set_experiment(cfg.mlflow_setup.experiment_name)

        with open(cfg.environment.mlflow.run_id_txt_path, 'r', encoding='utf-8') as f:
            parent_run_id = f.read()
        with mlflow.start_run(run_id=parent_run_id):
            return wrapped_function(*args, **kwargs)

    return wrapper


@app.command("trigger_experiment")
def trigger_experiment():
    '''
        Function to trigger an MLFLOW experiment
    '''
    mlflow.set_experiment(cfg.mlflow_setup.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow_setup.run_name):
        print(mlflow.active_run().info.run_id)

        with open(cfg.environment.mlflow.run_id_txt_path, 'w', encoding='utf-8') as f:
            f.write(mlflow.active_run().info.run_id)

        OmegaConf.save(cfg, 'experiment_config.yaml')
        mlflow.log_artifact(
            local_path='experiment_config.yaml',
            artifact_path=''
        )


@app.command("generate_datasets")
@mlflow_run
def generate_datasets() -> None:
    # Generate Data
    num_samples = 10_000  
    raw_data_df = pd.DataFrame({
        'numerical_1':    np.random.rand(num_samples),
        'numerical_2':    np.random.randn(num_samples),
        'numerical_3':    np.random.randint(0, 100, num_samples),
        
        'categorical_1':  np.random.choice(['x', 'y', 'z'], num_samples),
        'categorical_2':  np.random.choice(['N', 'S', 'E', 'W'], num_samples),
        
        'target':         np.random.randn(num_samples)
    })

    train_data_df, test_data_df = train_test_split(
        raw_data_df,
        test_size=cfg.sampling_strategy.train_test_split.test_size, 
        random_state=cfg.sampling_strategy.train_test_split.random_state
    )

    # Save Data into Cache
    train_data_df.to_parquet(cfg.environment.data_paths.raw_data_path + 'train_data.parquet', index=False)
    test_data_df.to_parquet(cfg.environment.data_paths.raw_data_path + 'test_data.parquet', index=False)


@app.command("log_training_dataset")
@mlflow_run
def log_training_dataset() -> None:
    # Log Training Data to MLFlow
    mlflow.log_artifact(
        local_path=cfg.environment.data_paths.raw_data_path + 'train_data.parquet',
        artifact_path=cfg.environment.artifact_paths.training_dataset_path.split(cfg.environment.artifact_paths.base_path)[-1]
    )


@app.command("preprocess_fit")
@mlflow_run
def preprocess_fit() -> None:
    train_data_df = pd.read_parquet(cfg.environment.data_paths.raw_data_path + 'train_data.parquet')

    # Fit Scaler
    standard_scaler = StandardScaler()
    numerical_columns_to_scale = [col.name for col in cfg.vars.num_vars if col.to_scale]
    standard_scaler.fit(train_data_df[numerical_columns_to_scale])

    # Save Scaler into Cache
    joblib.dump(standard_scaler, cfg.environment.artifacts_paths.scaler_path + 'standard_scaler.pkl')


@app.command("log_preprocessing_artifacts")
@mlflow_run
def log_preprocessing_artifacts() -> None:
    # Log Scaler to MLFlow
    mlflow.log_artifact(
        local_path=cfg.environment.artifacts_paths.scaler_path + 'standard_scaler.pkl',
        artifact_path=cfg.environment.artifact_paths.scaler_path.split(cfg.environment.artifact_paths.base_path)[-1]
    )


@app.command("preprocess_transform")
@mlflow_run
def preprocess_transform() -> None:
    train_data_df = pd.read_parquet(cfg.environment.data_paths.raw_data_path + 'train_data.parquet')
    test_data_df = pd.read_parquet(cfg.environment.data_paths.raw_data_path + 'test_data.parquet')

    # Preprocess Data
    numerical_columns_to_scale = [col.name for col in cfg.vars.num_vars if col.to_scale]
    categorical_columns_to_encode = [col.name for col in cfg.vars.cat_vars if col.to_encode]
    standard_scaler = joblib.load(cfg.environment.artifacts_paths.scaler_path + 'standard_scaler.pkl')
    
    train_data_df = preprocess_data(
        train_data_df,
        numerical_columns=numerical_columns_to_scale,
        categorical_columns=categorical_columns_to_encode,
        scaler=standard_scaler
    )
    test_data_df = preprocess_data(
        test_data_df,
        numerical_columns=numerical_columns_to_scale,
        categorical_columns=categorical_columns_to_encode,
        scaler=standard_scaler
    )

    # Save Preprocessed Data into Cache
    train_data_df.to_parquet(cfg.environment.data_paths.preprocessed_data_path + 'preprocessed_train_data.parquet', index=False)
    test_data_df.to_parquet(cfg.environment.data_paths.preprocessed_data_path + 'preprocessed_test_data.parquet', index=False)


@app.command("train_model")
@mlflow_run
def train_model() -> None:
    train_data_df = pd.read_parquet(cfg.environment.data_paths.preprocessed_data_path + 'preprocessed_train_data.parquet')

    # Train the Model
    params = dict(cfg.model.ridge)

    model = getattr(sklearn, cfg.model.sklearn_class(**params))
    
    X_train = train_data_df.drop(cfg.vars.target_var.name, axis=1)
    y_train = train_data_df[cfg.vars.target_var.name]

    model.fit(X_train, y_train)

    # Save Model into Cache
    joblib.dump(model, cfg.environment.artifacts_paths.model_path + 'ridge_regressor.pkl')

    # Register Model in MLFlow (required purely to enable promotion to production)
    from datetime import datetime
    mlflow.sklearn.log_model(  # For h20 models, use `mlflow.h20.log_model`
        sk_model=model,
        artifact_path='registered_model',
        registered_model_name=f'dummy_registered_model__{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    )


@app.command("log_model_artifacts")
@mlflow_run
def log_model_artifacts() -> None:
    # Log Model to MLFlow
    mlflow.log_artifact(
        local_path=cfg.environment.artifacts_paths.model_path + 'ridge_regressor.pkl',
        artifact_path=cfg.environment.artifact_paths.model_path.split(cfg.environment.artifact_paths.base_path)[-1]
    )


@app.command("evaluate_model")
@mlflow_run
def evaluate_model() -> None:
    test_data_df = pd.read_parquet(cfg.environment.data_paths.preprocessed_data_path + 'preprocessed_test_data.parquet')

    # Run Model on Test Data
    model = joblib.load(cfg.environment.artifacts_paths.model_path + 'ridge_regressor.pkl')

    X_test = test_data_df.drop(cfg.vars.target_var.name, axis=1)
    y_test = test_data_df[cfg.vars.target_var.name]

    y_pred = model.predict(X_test)

    # Calculate Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log Metrics to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)


if __name__ == "__main__":
    app()
