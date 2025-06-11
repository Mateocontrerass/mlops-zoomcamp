import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

url = 'http://localhost:9696/predict'
# Send a POST request to the Flask web service with the ride data
response = requests.post(url, json=ride)
print(response.json())
