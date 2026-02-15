import streamlit as st
import pandas as pd
import job pickle # or use your model loading library
import numpy as np

# Title of the app
st.title("Wellness Tourism Package Prediction")
st.markdown("Enter customer details to predict purchase likelihood.")

# Creating input fields for the user
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=25000)
duration = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)
passport = st.selectbox("Has Passport?", ["No", "Yes"])
past_trips = st.number_input("Number of Trips per Year", min_value=0, value=2)

# Convert inputs to a dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [income],
    'DurationOfPitch': [duration],
    'Passport': [1 if passport == "Yes" else 0],
    'NumberOfTrips': [past_trips]
})

# Note: In a real scenario, you'd load your model from HF here
# For the demo, we show the structure
if st.button("Predict"):
    # Simulated prediction logic based on model insights
    st.success("Prediction calculation successful! (Connect your HF Model for live results)")
