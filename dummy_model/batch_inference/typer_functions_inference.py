import os
import typer
import pandas as pd
from pathlib import Path
from pprint import pprint
from omegaconf import OmegaConf
from hydra import compose, initialize
from dictdiffer import diff as dictdiff

from ds_snowflake_utils import SnowflakeInterface

from dummy_model.modules import DUMMY_IMPORT__MODULES  # This import is only here to show you how you can import from the `modules` directory


with initialize(config_path='./config/', version_base=None):
    cfg = compose(config_name=f'{os.environ.get("DATALAB_DEPLOYMENT_ENVIRONMENT", "LOCAL")}')


snowflake_interface = SnowflakeInterface(
    stage_name_suffix=cfg['snowflake_staging']['stage_name_suffix'],
    s3_path_suffix=cfg['snowflake_staging']['s3_path_suffix']
)


app = typer.Typer(chain=True)


@app.command('test_command')
def test_command() -> None:
    #################################################
    print(f'Testing config read: {cfg["paths"]["raw_data_path"]}')
    #################################################
    
    #################################################
    print('Testing loading of artifacts:')
    #################################################
    import pickle
    for file in Path(cfg["paths"]["artifacts_path"]).rglob('*[pb][ki][ln]'):
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
    for path_config_key in cfg['paths']:
        if not os.path.exists(cfg['paths'][path_config_key]):
            raise ValueError(
                f'The {path_config_key} {cfg["paths"][path_config_key]} does not exist. \
                This is NOT oki doki.'
            )


@app.command("load_input")
def load_input() -> None:
    '''
        Function to load data from Snowflake
    '''
    print("Initiate Input Data Loading")

    ### REPLACE THIS ###
    raw_data_df = pd.DataFrame()
    ### REPLACE THIS ###

    raw_data_df.to_parquet(cfg['paths']['raw_data_path'] + 'raw_data.parquet')

    print("Input Data Loading successful!")


@app.command("preprocess_data")
def preprocess_data() -> None:
    '''
        Function to transform data using the preprocessing pipeline
    '''
    print("Initiate Data Preprocessing")
    
    raw_data_df = pd.read_parquet(cfg['paths']['raw_data_path'] + 'raw_data.parquet')

    ### REPLACE THIS ###
    preprocessed_data_df = raw_data_df.copy(deep=True)
    ### REPLACE THIS ###

    preprocessed_data_df.to_parquet(cfg['paths']['preprocessed_data_path'] + 'preprocessed_data.parquet')

    print("Data Preprocessing Successful!")


@app.command("run_inference")
def run_inference():
    '''
    Function to predict using the trained model
    '''
    print("Initiate Model Inference")

    preprocessed_data_df = pd.read_parquet(cfg['paths']['preprocessed_data_path'] + 'preprocessed_data.parquet')

    ### REPLACE THIS ###
    inferenced_data_df = preprocessed_data_df.copy(deep=True)
    ### REPLACE THIS ###

    inferenced_data_df.to_parquet(cfg['paths']['inferenced_data_path'] + 'inferenced_data.parquet')

    print("Model Inference Successful!")


@app.command("postprocess_data")
def postprocess_data():
    '''
    Function to post process output from the trained model
    '''
    print("Initiate Data Postprocessing")

    inferenced_data_df = pd.read_parquet(cfg['paths']['inferenced_data_path'] + 'inferenced_data.parquet')

    ### REPLACE THIS ###
    postprocessed_data_df = inferenced_data_df.copy(deep=True)
    ### REPLACE THIS ###

    postprocessed_data_df.to_parquet(cfg['paths']['postprocessed_data_path'] + 'postprocessed_data.parquet')

    print("Data Postprocessing Successful!")


@app.command("push_output")
def push_output() -> None:
    '''
        Function to push data to final output location
    '''
    print("Initiate Data Push")

    postprocessed_data_df = pd.read_parquet(cfg['paths']['postprocessed_data_path'] + 'postprocessed_data.parquet')

    ### REPLACE THIS ###
    pass
    ### REPLACE THIS ###

    print("Data Push Successful!")


if __name__ == "__main__":
    app()
