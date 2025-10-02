# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Set up the title and description
st.title('🏡 California House Price Prediction')
st.write('Enter the details of a house to get a price prediction.')

# Create input fields for the user
st.sidebar.header('Input Features')
med_inc = st.sidebar.slider('Median Income (in tens of thousands of $)', 0.5, 15.0, 8.0)
house_age = st.sidebar.slider('House Age (years)', 1.0, 55.0, 25.0)
ave_rooms = st.sidebar.slider('Average Number of Rooms', 1.0, 15.0, 5.0)
population = st.sidebar.slider('Population', 200.0, 40000.0, 1500.0)

# Create a button to make predictions
if st.sidebar.button('Predict Price'):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'Population': [population]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0] # Prediction is an array
    
    # Display the result
    st.subheader('Predicted Price')
    st.success(f'**${prediction * 100000:,.2f}**')
