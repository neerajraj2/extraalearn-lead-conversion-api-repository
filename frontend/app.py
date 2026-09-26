import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("ExtraaLearn Lead Conversion Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for lead features
age = st.number_input("Age", min_value=18, max_value=100, value=35)
current_occupation = st.selectbox("Current Occupation", ["Professional", "Unemployed", "Student"])
first_interaction = st.selectbox("First Interaction", ["Website", "Mobile App"])
profile_completed = st.selectbox("Profile Completed", ["High", "Medium", "Low"])
website_visits = st.number_input("Website Visits", min_value=0, value=3)
time_spent_on_website = st.number_input("Time Spent on Website (seconds)", min_value=0, value=700)
page_views_per_visit = st.number_input("Page Views per Visit", min_value=0.0, value=3.0, step=0.1)
last_activity = st.selectbox("Last Activity", ["Email Activity", "Phone Activity", "Website Activity"])
print_media_type1 = st.selectbox("Seen Ad in Newspaper (print_media_type1)?", ["No", "Yes"])
print_media_type2 = st.selectbox("Seen Ad in Magazine (print_media_type2)?", ["No", "Yes"])
digital_media = st.selectbox("Seen Ad on Digital Platforms?", ["No", "Yes"])
educational_channels = st.selectbox("Heard through Educational Channels?", ["No", "Yes"])
referral = st.selectbox("Heard through Referral?", ["No", "Yes"])

# Convert user input into a dictionary matching the model's expected input format
input_data = {
    'age': age,
    'current_occupation': current_occupation,
    'first_interaction': first_interaction,
    'profile_completed': profile_completed,
    'website_visits': website_visits,
    'time_spent_on_website': time_spent_on_website,
    'page_views_per_visit': page_views_per_visit,
    'last_activity': last_activity,
    'print_media_type1': print_media_type1,
    'print_media_type2': print_media_type2,
    'digital_media': digital_media,
    'educational_channels': educational_channels,
    'referral': referral,
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/lead", json=input_data)  # Send data to Flask API
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['Predicted Status Label']} (Conversion Probability: {result['Conversion Probability']})")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/leadbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
