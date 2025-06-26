"""Airflow DAG for data ingestion pipeline."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'customer_churn_data_ingestion',
    default_args=default_args,
    description='Daily data ingestion for customer churn prediction',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'data-ingestion', 'churn-prediction'],
)

def ingest_customer_profiles(**context):
    """Ingest customer profiles from PostgreSQL."""
    import sys
    import os
    sys.path.append('/opt/airflow/src')
    
    from src.data.ingestion import DataIngestionPipeline
    
    pipeline = DataIngestionPipeline()
    data = pipeline.ingest_customer_profiles()
    pipeline.save_raw_data(data, f"customer_profiles_{context['ds']}.parquet")
    
    return len(data)

def ingest_usage_logs(**context):
    """Ingest usage logs from S3."""
    import sys
    sys.path.append('/opt/airflow/src')
    
    from src.data.ingestion import DataIngestionPipeline
    
    pipeline = DataIngestionPipeline()
    data = pipeline.ingest_usage_logs(f"usage_logs/{context['ds']}/")
    
    if not data.empty:
        pipeline.save_raw_data(data, f"usage_logs_{context['ds']}.parquet")
        return len(data)
    
    return 0

def validate_ingested_data(**context):
    """Validate ingested data using Great Expectations."""
    import sys
    sys.path.append('/opt/airflow/src')
    
    from src.data.validation import DataValidator
    import pandas as pd
    from src.config.settings import settings
    
    validator = DataValidator()
    
    # Load today's data
    customer_file = settings.data_path / "raw" / f"customer_profiles_{context['ds']}.parquet"
    
    if customer_file.exists():
        customer_data = pd.read_parquet(customer_file)
        
        # Validate data
        data_dict = {"customer_profiles": customer_data}
        results = validator.validate_pipeline_data(data_dict)
        
        # Check if validation passed
        if not results.get("customer_profiles", {}).get("success", False):
            raise ValueError("Data validation failed")
        
        return results
    
    raise FileNotFoundError(f"Customer profiles file not found: {customer_file}")

def check_data_quality(**context):
    """Check data quality and completeness."""
    import sys
    sys.path.append('/opt/airflow/src')
    import pandas as pd
    from src.config.settings import settings
    
    # Load today's data
    customer_file = settings.data_path / "raw" / f"customer_profiles_{context['ds']}.parquet"
    
    if customer_file.exists():
        data = pd.read_parquet(customer_file)
        
        # Quality checks
        total_records = len(data)
        missing_values = data.isnull().sum().sum()
        duplicate_customers = data['customer_id'].duplicated().sum()
        
        quality_report = {
            "total_records": total_records,
            "missing_values": missing_values,
            "duplicate_customers": duplicate_customers,
            "missing_percentage": (missing_values / (total_records * len(data.columns))) * 100
        }
        
        # Fail if quality is too poor
        if quality_report["missing_percentage"] > 20:  # More than 20% missing
            raise ValueError(f"Data quality too poor: {quality_report['missing_percentage']:.2f}% missing")
        
        if duplicate_customers > 0:
            raise ValueError(f"Found {duplicate_customers} duplicate customers")
        
        return quality_report
    
    raise FileNotFoundError(f"Customer profiles file not found: {customer_file}")

# Task definitions
ingest_customers_task = PythonOperator(
    task_id='ingest_customer_profiles',
    python_callable=ingest_customer_profiles,
    dag=dag,
)

ingest_usage_task = PythonOperator(
    task_id='ingest_usage_logs',
    python_callable=ingest_usage_logs,
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_ingested_data,
    dag=dag,
)

check_quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

# Task to trigger feature engineering if data is good
trigger_feature_engineering = BashOperator(
    task_id='trigger_feature_engineering',
    bash_command='echo "Data ingestion completed successfully. Triggering feature engineering..."',
    dag=dag,
)

# Task dependencies
[ingest_customers_task, ingest_usage_task] >> validate_data_task >> check_quality_task >> trigger_feature_engineering
