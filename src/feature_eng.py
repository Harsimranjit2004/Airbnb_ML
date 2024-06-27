import numpy as np
import pandas as pd 
import sklearn
import joblib
import os
from sklearn.impute import SimpleImputer
from feature_engine.encoding import (RareLabelEncoder)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import ( OneHotEncoder,  FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
import warnings


pd.set_option("display.max_columns", None)
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


def load_data(train_path:str, test_path:str) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
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

def save_transformer(transformer: BaseEstimator, transformer_save_path: str) -> None:
    try:
        joblib.dump(transformer, transformer_save_path)
        print(f"Transformer saved to '{transformer_save_path}'")
    except Exception as e:
        print(f"Error: An error occurred while saving the transformer.")
        print(e)
        raise
def extract_date_features(df:pd.DataFrame)->pd.DataFrame:
    df['last_review'] = pd.to_datetime(df['last_review'])
    df['review_day'] = df['last_review'].dt.day
    df['review_month'] = df['last_review'].dt.month
    df['review_year'] = df['last_review'].dt.year
    
    df['is_weekend'] = df['last_review'].dt.dayofweek // 5
    return df[['review_day', 'review_month', 'review_year', 'is_weekend']]

def preFeatureEng(df:pd.DataFrame)->pd.DataFrame:
    try:
        return (
            df
            .drop(columns=['id', 'name', 'host_id', 'host_name', 'country', 'country_code', 'house_rules'])
            .assign(
                reviews_per_month=lambda df_: df_['reviews_per_month'].fillna(df_['reviews_per_month'].mean()),
                last_review=lambda df_: pd.to_datetime(df_['last_review']).interpolate(method='linear')
            )
            .dropna()
            .drop(columns=['cancellation_policy'])
        )
    except KeyError as e:
        print(f"Error: One or more columns to drop or transform are missing.")
        print(e)
        raise
    except Exception as e:
        print("Error: An unexpected error occurred during preprocessing.")
        print(e)
        raise


def applyFeatureEng(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        neighbourhood_group_transformer = Pipeline(steps=[
            ('grouper', RareLabelEncoder(tol=0.1, replace_with='other', n_categories=3)), 
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        neighbourhood_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        room_type_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])

        date_transformer = FunctionTransformer(extract_date_features, validate=False)

        last_review_transformer = Pipeline(steps=[
            ('extract_date', date_transformer)
        ])

        column_transformer = ColumnTransformer(transformers=[
            ("neighbourhood_group", neighbourhood_group_transformer, ['neighbourhood_group']),
            ('neighbourhood', neighbourhood_transformer, ['neighbourhood']),
            ('room_type', room_type_transformer, ['room_type']), 
            ('last_review', last_review_transformer, ['last_review'])
        ], 
        remainder='passthrough')

        train_transformed = column_transformer.fit_transform(train)
        save_transformer(column_transformer, 'models/preprocessor.joblib')
        test_transformed = column_transformer.transform(test)
        
        return train_transformed, test_transformed

    except KeyError as e:
        print(f"Error: One or more columns to transform are missing.")
        print(e)
        raise
    except Exception as e:
        print("Error: An unexpected error occurred during feature engineering.")
        print(e)
        raise
   


def save_data(train_processed:pd.DataFrame, test_processed:pd.DataFrame,data_path:str)->None:
    try:
        data_path = os.path.join(data_path, 'features')
        os.makedirs(data_path, exist_ok=True)
        train_processed.to_csv(os.path.join(data_path, 'train_features.csv'), index=False)
        test_processed.to_csv(os.path.join(data_path,'test_features.csv'),index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main()->None:
    train_data, test_data = load_data('./data/processed/train_processed.csv', './data/processed/test_processed.csv')

    train_data = preFeatureEng(train_data)
    test_data = preFeatureEng(test_data)

    train_transformed, test_transformed = applyFeatureEng(train_data, test_data)

    save_data(train_transformed, test_transformed, 'data')

if __name__ == "__main__":
    main()