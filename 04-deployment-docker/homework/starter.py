


# In[5]:


import pickle
import pandas as pd
import argparse


# In[6]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)




# In[7]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(year,month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='year YYYY')
    parser.add_argument('--month', type=int, required=True, help='month MM')

    args = parser.parse_args()

    df = read_data(args.year, args.month)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Standard Deviation: {y_pred.std()}')
    print(f'Mean: {y_pred.mean()}')
