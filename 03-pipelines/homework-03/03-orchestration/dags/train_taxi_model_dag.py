from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='train_taxi_model',
    default_args=default_args,
    schedule=None,  # You can change to '@monthly' later
    catchup=False,
    description='Train taxi model and log to MLflow',
) as dag:

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python3 /home/mateo//ml-projects/mlops-course/03-orchestration/scripts/train.py --year 2023 --month 3'
    )
