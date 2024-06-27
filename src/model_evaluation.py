import json
import pandas as pd
from sklearn.metrics import r2_score
import pickle

def load_data(test_path:str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(test_path)
        return test_data
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

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as e:
        print(f"Error: Model file not found at {model_path}")
        raise
    except Exception as e:
        print("Error: An unexpected error occurred while loading the model")
        print(e)
        raise

def calculate_metrics(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
        return r2
    except Exception as e:
        print("Error: An error occurred while calculating metrics")
        print(e)
        raise

def save_metrics(metrics, metrics_path: str):
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        print(f"Metrics saved to '{metrics_path}'")
    except Exception as e:
        print("Error: An error occurred while saving metrics")
        print(e)
        raise

def main():
    try:
        test_data_path = 'data/features/test_features.csv'
        test_data = load_data(test_data_path)

        model_path = 'models/xgb-model.pkl'
        model = load_model(model_path)

        X_test = test_data.drop(columns='remainder__price')
        y_test = test_data['remainder__price']

        y_pred = model.predict(X_test)

        r2 = calculate_metrics(y_test, y_pred)
        metrics = {'R2 Score': r2}
        metrics_path = 'metrics.json'
        save_metrics(metrics, metrics_path)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
