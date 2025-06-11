# 🐳 NYC Taxi Duration Prediction Service

This project demonstrates how to deploy a machine learning model as an online web service using Flask, Gunicorn, and Docker.  
It covers the full process: environment setup, model packaging, API creation, containerization, and testing.


[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://docs.docker.com/get-docker/)

## 📋 Project Structure

```

├── predict.py        # Main Flask application
├─── lin_reg.bin      # Trained ML model
├── test_flask.py # Integration tests
├── Dockerfile           # Container configuration
├── Pipfile             # Python dependencies
├── Pipfile.lock        # Locked dependencies
└── README.md          # This documentation
```



## 🚀 Quick Start

1. **Clone and Build:**
```bash
git clone https://github.com/yourusername/nyc-taxi-prediction
cd nyc-taxi-prediction
docker build -t taxi-prediction:v1 .
```

2. **Run the Service:**
```bash
docker run -d -p 9696:9696 --name taxi-predictor taxi-prediction:v1
```

3. **Test the API:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40}'
```


## 🔍 API Documentation

### Prediction Endpoint

**POST** `/predict`

Request Body:
```json
{
  "PULocationID": int,    // Pickup location ID
  "DOLocationID": int,    // Dropoff location ID
  "trip_distance": float  // Trip distance in miles
}
```

Response:
```json
{
  "prediction": float  // Predicted duration in minutes
}
```





## 🔧 Development Setup

### Prerequisites

- Docker Engine 24.0+
- Python 3.12+
- pipenv

### Local Development

1. **Set up environment:**
````bash
# Install dependencies
pipenv install

# Activate virtual environment
pipenv shell

# Run Flask development server
python app/predict.py
````


## Model Preparation

1. **Check scikit-learn version used for your model:**

```bash
pip freeze | grep scikit-learn
# Example output: scikit-learn==1.6.1
```

2. **Install dependencies in a virtual environment:**

```bash
sudo apt install pipenv  # If pipenv is not installed
pipenv install scikit-learn==1.6.1 flask
```

> **Note:**  
> If you already have a virtual environment activated, `pipenv shell` will spawn a new shell inside the pipenv environment.  
> If you are inside another venv, deactivate it first to avoid conflicts.

3. **Activate the environment:**

```bash
pipenv shell
```

---

## Building the Prediction Service

1. **Create `predict.py`**  
   This script loads your trained model and exposes a prediction API.

2. **Key functions:**
    - `prepare_data`: Prepares input data for the model.
    - `predict`: Runs the model prediction.

3. **Create the Flask app:**

```python
app = Flask('duration-prediction')
```

4. **Add the prediction endpoint:**

```python
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_data(ride)
    pred = predict(features)
    return jsonify({'prediction': pred})
```

**Why do we need this decorator?**  
`@app.route('/predict', methods=['POST'])` tells Flask to listen for POST requests at the `/predict` URL and call the `predict_endpoint` function when such a request is received. This is how we expose our model as a web API.

5. **Run the Flask app:**

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```

**Why is this needed?**  
This block ensures the Flask app only runs when the script is executed directly (not when imported as a module).  
`host='0.0.0.0'` makes the service accessible from outside the container or VM, and `port=9696` sets the listening port.

---

## Testing Locally

1. **Start the Flask app:**

```bash
python predict.py
```

2. **Create `test_flask.py` to test the endpoint:**

```python
import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())
```

3. **Run the test:**

```bash
python test_flask.py
```

You should see a JSON response with the prediction.

---

## Production-Ready with Gunicorn

Instead of running Flask directly (which is not recommended for production), use Gunicorn:

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

- **Why use Gunicorn?**  
  Gunicorn is a production-grade WSGI server that handles multiple requests concurrently and provides better performance than Flask's development server.

- Make sure to install Gunicorn in your virtual environment:
```bash
pipenv install gunicorn
```

---

## Containerizing with Docker

1. **Create a `Dockerfile`:**

```dockerfile
FROM python:3.12.3-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn","--bind=0.0.0.0:9696", "predict:app" ]
```

2. **Build the Docker image:**

```bash
docker build -t ride-duration-prediction-service:v1 .
```

The `.` at the end tells Docker to use the current directory as the build context.

---

## Testing the Dockerized Service

1. **Run the container:**

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```

**Command flags explained:**
- `-it`: Interactive mode (enables Ctrl+C to stop)
- `--rm`: Remove container after stopping
- `-p 9696:9696`: Map host port to container port

2. **Test the containerized service:**

```bash
python test_flask.py
```

The test should work exactly as before, but now it's hitting the containerized service.

---

## Recommendations & Notes

- **Version Control:** Always specify exact versions in `Pipfile` to ensure reproducibility
- **Environment Variables:** Use them for configuration in production
- **Logging:** Add proper logging in production deployments
- **Health Checks:** Add an endpoint to monitor service health
- **Security:** Review Flask security best practices
- **Resource Limits:** Set container memory/CPU limits in production

---

## What I Learned

- Building production-ready ML web services
- Using Flask for API development
- Containerization with Docker
- Environment isolation with pipenv
- Production deployment considerations








