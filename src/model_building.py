import numpy as np
import pandas as pd
import pickle
import os
from xgboost import XGBRegressor

def load_data(train_path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(train_path)
        return train_data
    except pd.errors.ParserError as e:
        print(f'Error: failed to parse the CSV file(s)')
        print(e)
        raise
    except FileNotFoundError as e:
        print(f'Error: file not found')
        print(e)
        raise
    except Exception as e:
        print("Error: An unexpected error occurred while loading the data")
        print(e)
        raise

def train_model(train_data: pd.DataFrame) -> XGBRegressor:
    try:
        X_train = train_data.drop(columns='remainder__price')
        y_train = train_data['remainder__price'].copy()

        xgb_model = XGBRegressor(random_state=42,
                                 colsample_bytree=0.9,
                                 learning_rate=0.1,
                                 max_depth=4,
                                 min_child_weight=5,
                                 n_estimators=100,
                                 subsample=0.9)

        xgb_model.fit(X_train, y_train)

        return xgb_model
    except Exception as e:
        print(f"Error: An error occurred during model training.")
        print(e)
        raise

def save_model(model, model_save_path: str):
    try:
        model_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        with open(model_save_path, 'wb') as file:
            pickle.dump(model, file)

        print(f"XGBoost model saved to '{model_save_path}'")
    except Exception as e:
        print(f"Error: An error occurred while saving the model.")
        print(e)
        raise

def main():
    try:
        train_data_path = './data/features/train_features.csv'

        train_data = load_data(train_data_path)
        xgb_model = train_model(train_data)

        model_save_path = 'models/xgb-model.pkl'
        save_model(xgb_model, model_save_path)
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
