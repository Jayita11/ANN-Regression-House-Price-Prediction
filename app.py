import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")

# Load the pre-trained model
model = tf.keras.models.load_model('model_ann_reg.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Load the scaler
with open('min_max_scaler.pkl', 'rb') as file:
    min_max_scaler = pickle.load(file)

# Streamlit app
st.title('House Price Prediction')

# User input
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.slider('Housing Median Age', 1, 100, 30)
total_rooms = st.number_input('Total Rooms')
total_bedrooms = st.number_input('Total Bedrooms')
population = st.number_input('Population')
households = st.number_input('Households')
median_income = st.number_input('Median Income')
ocean_proximity = st.selectbox('Ocean Proximity', ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])
ocean_proximity_mapping = {"<1H OCEAN": 0, "INLAND": 1, "NEAR OCEAN": 2, "NEAR BAY": 3, "ISLAND": 4}
ocean_proximity = ocean_proximity_mapping[ocean_proximity]

# Prepare the input data
input_data = np.array([longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity])
input_data_scaled = min_max_scaler.transform([input_data])

# Predict house price
prediction = model.predict(input_data_scaled)
predicted_price = prediction[0][0]

st.write(f'Predicted House Price: ${predicted_price:.2f}')
