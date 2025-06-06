
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import mlflow
import pickle
import os
import pyarrow.parquet as pq
import pyarrow.dataset as ds

def read_dataframe(year, month):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    df = pd.read_parquet(url)
    df = df.head(10000)

    print(f'Loaded {len(df)} rows from {url}')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f'Resulting rows after filtering: {len(df)}')

    return df
    

def train_and_log_model(df):   
    dv = DictVectorizer()
    lr = LinearRegression()

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X = dv.fit_transform(dicts)
    y = df.duration.values
    lr.fit(X, y)

    y_pred = lr.predict(X)
    rmse = mean_squared_error(y, y_pred)**(1/2)

    print(f'Intercept: {lr.intercept_}')
    print(f'RMSE: {rmse}')

    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('nyc-taxi-experiment')

    with mlflow.start_run():

        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("vectorizer", "DictVectorizer")
        mlflow.log_metric('rmse', rmse)

        # save model to a local folder
        os.makedirs('mlflow_artifact', exist_ok=True)
        model_path = 'mlflow_artifact/model.pkl'
        vd_path = 'mlflow_artifact/dv.pkl'

        with open(model_path, 'wb') as f_out:
            pickle.dump(lr, f_out)
        with open(vd_path, 'wb') as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(vd_path, artifact_path="dv.pkl")



    print('Model and artifacts logged to MLflow')

    return dv, lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load data and train model')
    parser.add_argument('--year', type=int, default=2023, help='Year of the data to load')
    parser.add_argument('--month', type=int, default=3, help='Month of the data to load')
    args = parser.parse_args()


    df = read_dataframe(args.year, args.month)
    dv, lr = train_and_log_model(df)
    print('Training completed successfully.')

