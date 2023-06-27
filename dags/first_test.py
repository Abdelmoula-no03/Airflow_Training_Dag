from datetime import datetime,timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
#from airflow.models import Xcom
from helpers import extract_clean, train_test_model
from helpers import train_test_model


def extract_clean_test():

    cleaned_data = extract_clean()
    #context['ti'].xcom_push(key ='cleaned_data',value = cleaned_data)
    print("extraction et nettoyage des données passé avec succés")
    return cleaned_data

    #print("positive reviews :" ,df[df['label']==1].shape[0])
    #print("negative reviews :" ,df[df['label']==0].shape[0])

def train_test(ti):

    cleaned_data  = ti.xcom_pull(task_ids = 'extract_clean')
    #print('la dimension des données est :', cleaned_data.shape)
    ancien_acc, new_acc = train_test_model(cleaned_data)
    print('le modèle a été réentrainé et testé avec succés')
    print("Ancien Accuracy :", ancien_acc)
    print("la nouvelle Accuracy :", new_acc)

    




    

default_args = {
    'owner': 'abdelmoula',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'extract_clean_train_dag_lastestV.1',
    default_args=default_args,
    schedule_interval= '@daily'
    

) as dag:

  extract_clean_task = PythonOperator(
      task_id='extract_clean',
      python_callable=extract_clean_test,
      dag=dag    
)

  train_test_task = PythonOperator(
     task_id = 'train_test',
     python_callable = train_test,
     dag=dag
  )
  
# mettre les dependances entre les taches :

  extract_clean_task >> train_test_task
 
