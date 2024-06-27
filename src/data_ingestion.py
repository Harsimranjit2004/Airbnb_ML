from dotenv import load_dotenv
import os
import shutil
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_params(params_path:str)->float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size= params['data_ingestion']['test_size']
        return test_size
    except FileNotFoundError:
        print('File not found Error')
        raise
    except yaml.YAMLError as e:
        print('Yaml file Error')
        print(e)
        raise
    except Exception as e:
        print('some error occured in loading params file')
        print(e)
        raise

def create_kaggle_json(username, key):
    home = str(Path.home())
    kaggle_dir = os.path.join(home, '.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')

    try:
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_file, 'w') as f:
            f.write('{{"username":"{}","key":"{}"}}\n'.format(username, key))
        print(f'Created {kaggle_file}')

    except OSError as e:
        print(f'Error: Failed to create directory or file in {kaggle_dir}')
        print(e)
        raise
    except IOError as e:
        print(f'Error: Failed to open or write to file {kaggle_file}')
        print(e)
        raise
    except Exception as e:
        print("Error: An unexpected error occurred while creating kaggle.json")
        print(e)
        raise
def download_from_kaggle(datasetName:str, filePath:str)->None:
    import kaggle

    try:
        load_dotenv()
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')

        if kaggle_username is None or kaggle_key is None: 
            raise ValueError("KAGGLE_USERNAME or KAGGLE_KEY environment variables not found. Please set them in your .env file.")
        
        create_kaggle_json(kaggle_username, kaggle_key)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(datasetName, path=filePath, unzip=True)
        print(f"Dataset '{datasetName}' downloaded successfully to '{filePath}'")

    except kaggle.exceptions.KaggleApiError as e: 
        print(f"Error: Kaggle API returned an error while downloading '{datasetName}'")
        print(e)
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except FileNotFoundError as e:
        print("Error: The .env file was not found.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while downloading '{datasetName}'")
        print(e)
        raise

def load_data(data_path:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except pd.errors.ParserError as e: 
        print(f'Error: failed to parse the CSV file form {data_path}')
        print(e)
        raise
    except Exception as e: 
        print("Error: An unexpected error occured while loading the data")
        print(e)
        raise

def save_data(train_data: pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None: 
    try:
        data_path = os.path.join(data_path, 'raw')
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"),index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main()->None:
    try: 
        test_size = load_params(params_path='params.yaml')
        dataset_name = 'arianazmoudeh/airbnbopendata' 
        download_path = 'data/raw'
        download_from_kaggle(dataset_name, download_path)
        df = load_data('/Users/harsimranjitsingh/Desktop/Airbnb_Project/data/raw/Airbnb_Open_Data.csv')
        train_data, test_data = train_test_split(df, test_size=test_size,random_state=42)
        save_data(train_data, test_data, data_path='data')
    except Exception as e: 
        print(f"Failed to complete the data ingestion process.")
        print(e)
        
if __name__ == "__main__":
    main()
    