# File to load the model and make predictions
import os
import pickle
import numpy as np

from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f:
    (dv, model) = pickle.load(f)

def prepare_data(data):
    features = {}
    features["PU_DO"] = '%s_%s' % (data['PULocationID'], data['DOLocationID'])
    features["trip_distance"] = data['trip_distance']
    return features


def predict(data):
    """
    Predicts the target variable for the given data.
    
    Args:
        data (dict): Input data for prediction.
        
    Returns:
        list: Predicted values.
    """
    X = dv.transform(data)
    y_pred = model.predict(X)
    return y_pred[0]


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])

def predict_endpoint():
    """
    Flask endpoint to handle prediction requests.
    
    Returns:
        JSON response with the prediction result.
    """
    ride = request.get_json()
    features = prepare_data(ride)
    pred = predict(features)

    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    
# This script implements a Flask web service for predicting ride durations using a pre-trained model.
# It loads the model and DictVectorizer, prepares incoming data, and exposes a '/predict' endpoint.
# The service accepts POST requests with ride data, processes the input, and returns the prediction as JSON.
# Designed for standalone execution, it runs the Flask app on port 9696.      
