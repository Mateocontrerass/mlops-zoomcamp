#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope 

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'

    df = pd.read_parquet(url)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_x(df , dv=None):

    categorical = ['PU_DO'] 
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:

        dv = DictVectorizer(sparse=True)
        x = dv.fit_transform(dicts)
    else:
        x = dv.transform(dicts)
 
    return x,dv

def train_model(x_train, y_train, x_val, y_val, dv):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        train = xgb.DMatrix(x_train, label=y_train)
        valid = xgb.DMatrix(x_val, label=y_val)

        best_params = {
            'max_depth': 84,
            'learning_rate': 0.06248648809504339,
            'reg_alpha' : 0.01868809980419891,
            'reg_lambda': 0.08788616023689615,
            'min_child_weight': 3.4431892558874027,
            'objective' : 'reg:linear',
            'seed' : 123
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=15,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run_id  # Optional: if you want to use it later


def run(year,month):
    df_train = read_dataframe(year=year, month=month)
    
    next_year = year if month<12 else year+1
    next_month = month+1 if month<12 else 1

    df_val = read_dataframe(year=next_year , month=next_month)

    x_train , dv = create_x(df_train)
    x_val ,_ = create_x(df_val,dv)

    target = 'duration'

    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(x_train, y_train, x_val, y_val, dv)
    
    # Optional: use the run_id here (e.g., print it, register model, etc.)
    print(f"Model training completed. MLflow Run ID: {run_id}")

if __name__ == "__main__":
    # Use argparse to get year and month frm command line
    import argparse

    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration.")
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train')
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    run(year = args.year, month= args.month)

