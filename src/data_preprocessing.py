import numpy as np 
import pandas as pd 
import os 

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


def Clean_data(df:pd.DataFrame)-> pd.DataFrame:
    return (
        df
        .drop_duplicates()
        .drop('license', axis=1)
        .rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
        .assign(
            name = lambda df_ : (
                df_
                .name
                .str
                .lower()
                .str.strip()
                .str.replace(r'[^\w\s]', '', regex=True)
                .str.replace(r'\d+', '', regex=True)
                .str.replace(r'\s+', '_', regex=True)
                .fillna('unknown')
            ), 
            host_identity_verified = lambda df_ : (
                df_
                .host_identity_verified
                .fillna('unconfirmed')
                .map({'unconfirmed': 0, 'verified': 1})
            ),
            neighbourhood_group = lambda df_ : (
                df_
                .neighbourhood_group
                .replace('Brooklyn', 'brookln')
                .replace('Manhattan','manhatan')
            ),
            country = lambda df_ : (
                df_ 
                .country
                .fillna('United States')
            ), 
            country_code = lambda df_ : (
                df_ 
                .country_code
                .fillna('US')
            ),
            instant_bookable = lambda df_: (
                df_
                .instant_bookable
                .astype(float)
            ),
            construction_year = lambda df_ : (
                df_ 
                .construction_year
                .astype(float)
            ),
            price = lambda df_ : (
                df_
                .price
                .fillna('0')
                .str.replace('[\$,]', '', regex=True)
                .astype('float64')
            ),
            service_fee = lambda df_ : (
                df_ 
                .service_fee
                .fillna('0')
                .str.replace('[\$,]', '', regex=True)
                .astype('float64')
            ),
            minimum_nights = lambda df_ : (
                df_ 
                .minimum_nights
                .astype(float)
                .where(df_['minimum_nights'] >= 0, 0)
            ),
            number_of_reviews = lambda df_ : (
                df_
                .number_of_reviews 
                .astype(float)
            ), 
            last_review = lambda df_ : (
                pd.to_datetime(df_['last_review'], errors='coerce')
            ),
            availability_365 = lambda df_ : (
                df_ 
                .availability_365 
                .where(df_['availability_365'] >=0, 0)
            )
        ) 
    )

def save_data(train_processed:pd.DataFrame, test_processed:pd.DataFrame,data_path:str)->None:
    try:
        data_path = os.path.join(data_path, 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def main()->None:
    train_data, test_data = load_data('./data/raw/train.csv', './data/raw/test.csv')

    train_processed_data = Clean_data(train_data)
    test_processed_data = Clean_data(test_data)

    save_data(train_processed_data, test_processed_data,'data')

if __name__ == "__main__":
    main()