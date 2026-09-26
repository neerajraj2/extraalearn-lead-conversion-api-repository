# Import necessary libraries
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
lead_conversion_predictor_api = Flask("ExtraaLearn Lead Conversion Predictor")

# Load the trained machine learning model (pipeline includes preprocessing)
model = joblib.load("extraalearn_lead_conversion_model_v1_0.joblib")


# Define a route for the home page (GET request)
@lead_conversion_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the ExtraaLearn Lead Conversion Prediction API!"


# Define an endpoint for single-lead prediction (POST request)
@lead_conversion_predictor_api.post('/v1/lead')
def predict_lead_conversion():
    """
    This function handles POST requests to the '/v1/lead' endpoint.
    It expects a JSON payload containing a single lead's details and returns
    the predicted conversion status (and probability of conversion) as a JSON response.
    """
    # Get the JSON data from the request body
    lead_data = request.get_json()

    # Extract the relevant features from the JSON data, mirroring the columns used during training
    sample = {
        'age': lead_data['age'],
        'current_occupation': lead_data['current_occupation'],
        'first_interaction': lead_data['first_interaction'],
        'profile_completed': lead_data['profile_completed'],
        'website_visits': lead_data['website_visits'],
        'time_spent_on_website': lead_data['time_spent_on_website'],
        'page_views_per_visit': lead_data['page_views_per_visit'],
        'last_activity': lead_data['last_activity'],
        'print_media_type1': lead_data['print_media_type1'],
        'print_media_type2': lead_data['print_media_type2'],
        'digital_media': lead_data['digital_media'],
        'educational_channels': lead_data['educational_channels'],
        'referral': lead_data['referral'],
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction (0 = will not convert, 1 = will convert)
    predicted_class = int(model.predict(input_data)[0])

    # Predicted probability of conversion (class 1)
    predicted_probability = round(float(model.predict_proba(input_data)[0][1]), 4)

    # Return the prediction and the probability of conversion
    return jsonify({
        'Predicted Status': predicted_class,
        'Predicted Status Label': 'Likely to Convert' if predicted_class == 1 else 'Not Likely to Convert',
        'Conversion Probability': predicted_probability,
    })


# Define an endpoint for batch prediction (POST request)
@lead_conversion_predictor_api.post('/v1/leadbatch')
def predict_lead_conversion_batch():
    """
    This function handles POST requests to the '/v1/leadbatch' endpoint.
    It expects a CSV file containing details for multiple leads and returns
    the predicted conversion statuses as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all leads in the DataFrame
    predicted_classes = model.predict(input_data).tolist()
    predicted_probabilities = model.predict_proba(input_data)[:, 1].tolist()

    # Create a dictionary of predictions with lead IDs as keys
    lead_ids = input_data['ID'].tolist() if 'ID' in input_data.columns else list(range(len(input_data)))
    output_dict = {
        str(lead_id): {
            'Predicted Status': int(pred_class),
            'Conversion Probability': round(float(pred_prob), 4),
        }
        for lead_id, pred_class, pred_prob in zip(lead_ids, predicted_classes, predicted_probabilities)
    }

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    lead_conversion_predictor_api.run(debug=True)
