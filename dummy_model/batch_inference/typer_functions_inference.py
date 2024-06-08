import os
from pathlib import Path
from pprint import pprint
from datetime import datetime

import typer
import joblib
import pandas as pd
from omegaconf import OmegaConf
from hydra import compose, initialize
from dictdiffer import diff as dictdiff

from ds_snowflake_utils import SnowflakeInterface

from dummy_model.modules.preprocessing import encode_and_scale_data


with initialize(config_path='./config/', version_base=None):
    cfg = compose(config_name=f'{os.environ.get("DATALAB_DEPLOYMENT_ENVIRONMENT", "LOCAL")}')


snowflake_interface = SnowflakeInterface(
    model='dummy_model',
    environment=os.environ.get('DATALAB_DEPLOYMENT_ENVIRONMENT', 'LOCAL')
)


app = typer.Typer(chain=True)


@app.command('test_command')
def test_command() -> None:
    #################################################
    print(f'Testing config read: {cfg.environment.data_paths.raw_data_path}')
    #################################################
    
    #################################################
    print('Testing loading of artifacts:')
    #################################################
    import pickle
    for file in Path(cfg.environment.artifact_paths.base_path).rglob('*[pb][ki][ln]'):
        with open(file, 'rb') as f:
            print(f'Loading {file}...')
            dummy_model_object = pickle.load(f)
            print(f'{file} loaded successfully\n')
            del dummy_model_object

    #################################################
    print('Testing consistency of configs: ')
    #################################################
    def _check_config_consistency(config_A: str = None, config_B: str = None):
        with initialize(config_path='./config/', version_base=None):
            config_dict_A = OmegaConf.to_container(compose(config_name=config_A))
        with initialize(config_path='./config/', version_base=None):
            config_dict_B = OmegaConf.to_container(compose(config_name=config_B))
        
        diff = dictdiff(config_dict_A, config_dict_B)
        print(f'Diff between {config_A} and {config_B} configs: ')
        pprint(list([d for d in diff if d[0] in ['add', 'remove']]))

        diff = dictdiff(config_dict_A, config_dict_B)
        for d in diff:
            operation = d[0]
            if operation in ['add', 'remove']:
                raise ValueError(f'{config_A} and {config_B} configs do not match.')

        print(f'{config_A} and {config_B} configs are consistent')

    _check_config_consistency('LOCAL'  ,   'DEV' )
    _check_config_consistency('DEV'    ,   'UAT' )
    _check_config_consistency('UAT'    ,   'PROD')


    #################################################
    print('Testing validity of paths in config: ')
    #################################################
    for path_config_key in cfg.environment.data_paths:
        if not os.path.exists(cfg.environment.data_paths[path_config_key]):
            raise ValueError(
                f'The {path_config_key} {cfg.environment.data_paths[path_config_key]} does not exist. \
                This is NOT oki doki.'
            )


@app.command("load_input")
def load_input() -> None:
    '''
        Function to load data from Snowflake
    '''
    print("Initiate Input Data Loading")

    raw_data_df = snowflake_interface.read_data(
        sql_query='SELECT * FROM DUMMY_MODEL_INFERENCE_INPUT',
        database='ANALYTICS_PROD',
        schema='DATA_SCIENCE_DATA_SOURCES'
    )

    raw_data_df.to_parquet(cfg.environment.data_paths.raw_data_path + 'raw_data.parquet')

    print("Input Data Loading successful!")


@app.command("preprocess_data")
def preprocess_data() -> None:
    '''
        Function to transform data using the preprocessing pipeline
    '''
    print("Initiate Data Preprocessing")
    
    raw_data_df = pd.read_parquet(cfg.environment.data_paths.raw_data_path + 'raw_data.parquet')

    numerical_columns_to_scale = [col.name for col in cfg.vars.num_vars if col.to_scale]
    categorical_columns_to_encode = [col.name for col in cfg.vars.cat_vars if col.to_one_hot_encode]
    print(categorical_columns_to_encode)
    standard_scaler = joblib.load(cfg.environment.artifact_paths.scaler_path + 'standard_scaler.pkl')
    
    preprocessed_data_df = encode_and_scale_data(
        data=raw_data_df,
        numerical_columns=numerical_columns_to_scale,
        categorical_columns=categorical_columns_to_encode,        
        scaler=standard_scaler
    )

    preprocessed_data_df.to_parquet(cfg.environment.data_paths.preprocessed_data_path + 'preprocessed_data.parquet')

    print("Data Preprocessing Successful!")


@app.command("run_inference")
def run_inference():
    '''
    Function to predict using the trained model
    '''
    print("Initiate Model Inference")

    preprocessed_data_df = pd.read_parquet(cfg.environment.data_paths.preprocessed_data_path + 'preprocessed_data.parquet')

    model = joblib.load(cfg.environment.artifact_paths.model_path + 'ridge_regressor.pkl')

    prediction = model.predict(preprocessed_data_df.drop('ID', axis=1))
    inferenced_data_df = preprocessed_data_df.copy(deep=True)
    inferenced_data_df['prediction'] = prediction

    inferenced_data_df.to_parquet(cfg.environment.data_paths.inferenced_data_path + 'inferenced_data.parquet')

    print("Model Inference Successful!")


@app.command("postprocess_data")
def postprocess_data():
    '''
    Function to post process output from the trained model
    '''
    print("Initiate Data Postprocessing")

    inferenced_data_df = pd.read_parquet(cfg.environment.data_paths.inferenced_data_path + 'inferenced_data.parquet')

    postprocessed_data_df = inferenced_data_df.copy(deep=True)
    postprocessed_data_df['inference_timestamp'] = datetime.now()

    postprocessed_data_df.to_parquet(cfg.environment.data_paths.postprocessed_data_path + 'postprocessed_data.parquet')

    print("Data Postprocessing Successful!")


@app.command("push_output")
def push_output() -> None:
    '''
        Function to push data to final output location
    '''
    print("Initiate Data Push")

    postprocessed_data_df = pd.read_parquet(cfg.environment.data_paths.postprocessed_data_path + 'postprocessed_data.parquet')

    snowflake_interface.push_data(
        df=postprocessed_data_df,
        schema='DATA_SCIENCE_OUTPUTS',
        table='DUMMY_MODEL_INFERENCE_OUTPUT',
        overwrite_table=False
    )

    print("Data Push Successful!")


if __name__ == "__main__":
    app()
