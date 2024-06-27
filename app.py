import numpy as np
import streamlit as st
import pandas as pd
import joblib
import pickle
import xgboost as xgb
####################### Other Helping Functions ###########################


def extract_date_features(df:pd.DataFrame)->pd.DataFrame:
    df['last_review'] = pd.to_datetime(df['last_review'])
    df['review_day'] = df['last_review'].dt.day
    df['review_month'] = df['last_review'].dt.month
    df['review_year'] = df['last_review'].dt.year
    
    df['is_weekend'] = df['last_review'].dt.dayofweek // 5
    return df[['review_day', 'review_month', 'review_year', 'is_weekend']]

def preFeatureEng(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = (
            df.drop(columns=['id', 'name', 'host_id', 'host_name', 'country', 'country_code', 'house_rules'])
            .assign(
                reviews_per_month=lambda df_: df_['reviews_per_month'].fillna(df_['reviews_per_month'].mean()),
                last_review=lambda df_: pd.to_datetime(df_['last_review']).interpolate(method='linear')
            )
            .dropna()
            .drop(columns=['cancellation_policy'])
        )
        return df
    except KeyError as e:
        st.error(f"Error: One or more columns to drop or transform are missing. {e}")
        raise
    except Exception as e:
        st.error(f"Error: An unexpected error occurred during preprocessing. {e}")
        raise

####################### Streamlit App ###########################

train = pd.read_csv('./data/processed/train_processed.csv')
st.title('Airbnb Price Prediction')
user_input = {}

user_input['host_identify_verified'] = st.selectbox(
    "Host Identity Verified", 
    options=['verified', 'unconfirmed']
)

user_input['neighbourhood_group'] = st.selectbox(
    'Neighbourhood Group',
    options=train.neighbourhood_group.dropna().unique()
)

user_input['neighbourhood'] = st.selectbox(
    'Neighbourhood',
    options=train.neighbourhood.dropna().unique()
)

user_input['lat'] = st.slider(
    'Select Latitude',
    min_value=train.lat.min(),
    max_value=train.lat.max()
)

user_input['long'] = st.slider(
    'Select Longitude',
    min_value=train.long.min(),
    max_value=train.long.max()
)

user_input['instant_bookable'] = st.radio(
    'Select Instant Bookable Option',
    options=[True, False]
)

user_input['room_type'] = st.selectbox(
    'Select Room Type',
    options=train.room_type.unique()
)

user_input['service_fee'] = st.number_input(
    'Service Fees',
    step=10,
    min_value=0
)

user_input['minimum_nights'] = st.number_input(
    'Minimum Number of Nights',
    step=1,
    min_value=0
)

user_input['number_of_reviews'] = st.number_input(
    'Number of Reviews',
    step=1,
    min_value=0
)


x_new = pd.DataFrame(user_input, index=[0])


for col in train.columns:
    if col not in x_new.columns:
        x_new[col] = np.nan


if st.button("Predict"):
    try:
        preprocessor_path = "/Users/harsimranjitsingh/Desktop/Airbnb_Project/models/preprocessor.joblib"
        saved_preprocessor = joblib.load(preprocessor_path)
        data = saved_preprocessor.transform(x_new)

        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 1:]  

        elif isinstance(data, np.ndarray):
            data = data[:, 1:]  
        model_path = "/Users/harsimranjitsingh/Desktop/Airbnb_Project/models/xgb-model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        x_new_xgb = xgb.DMatrix(data)
        pred = model.predict(data)[0]
        st.info(f"The predicted price is {pred:,.0f} INR")
    except Exception as e:
        st.error(f"Error during preprocessing or prediction: {e}")