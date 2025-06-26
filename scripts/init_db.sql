-- Initialize database schema for customer churn prediction

-- Create database extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Customer profiles table
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id VARCHAR(50) PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    tenure_months INTEGER,
    contract_type VARCHAR(20),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(12,2),
    payment_method VARCHAR(30),
    internet_service VARCHAR(20),
    phone_service VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Usage logs table
CREATE TABLE IF NOT EXISTS usage_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
    date DATE,
    data_usage_gb DECIMAL(10,3),
    call_minutes INTEGER,
    sms_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Support tickets table
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id VARCHAR(50) REFERENCES customer_profiles(customer_id),
    category VARCHAR(50),
    priority VARCHAR(20),
    status VARCHAR(20),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction logs table for monitoring
CREATE TABLE IF NOT EXISTS prediction_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id VARCHAR(50),
    prediction INTEGER,
    probability DECIMAL(5,4),
    latency_ms DECIMAL(8,2),
    model_metadata JSONB,
    batch_size INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    model_source VARCHAR(20)
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value DECIMAL(10,6),
    evaluation_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data drift monitoring table
CREATE TABLE IF NOT EXISTS drift_monitoring (
    drift_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_name VARCHAR(100),
    drift_score DECIMAL(6,4),
    drift_method VARCHAR(50),
    drift_detected BOOLEAN,
    reference_period_start DATE,
    reference_period_end DATE,
    current_period_start DATE,
    current_period_end DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlflow_run_id VARCHAR(100),
    model_name VARCHAR(100),
    model_type VARCHAR(50),
    training_data_size INTEGER,
    features_count INTEGER,
    roc_auc DECIMAL(6,4),
    accuracy DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    training_duration_minutes INTEGER,
    trigger_reason VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customer_profiles_updated_at ON customer_profiles(updated_at);
CREATE INDEX IF NOT EXISTS idx_usage_logs_customer_date ON usage_logs(customer_id, date);
CREATE INDEX IF NOT EXISTS idx_support_tickets_customer ON support_tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON prediction_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_customer ON prediction_logs(customer_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics_name_version ON model_metrics(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_drift_monitoring_feature ON drift_monitoring(feature_name);
CREATE INDEX IF NOT EXISTS idx_training_runs_created_at ON training_runs(created_at);

-- Insert sample data for testing
INSERT INTO customer_profiles (customer_id, age, gender, tenure_months, contract_type, monthly_charges, total_charges, payment_method, internet_service, phone_service) VALUES
('CUST_001', 34, 'F', 24, 'Two year', 79.99, 1919.76, 'Credit card', 'Fiber optic', 'Yes'),
('CUST_002', 45, 'M', 12, 'Month-to-month', 65.50, 786.00, 'Bank transfer', 'DSL', 'No'),
('CUST_003', 28, 'F', 6, 'One year', 89.99, 539.94, 'Electronic check', 'Fiber optic', 'Yes'),
('CUST_004', 52, 'M', 36, 'Two year', 55.00, 1980.00, 'Credit card', 'DSL', 'Yes'),
('CUST_005', 31, 'F', 3, 'Month-to-month', 95.00, 285.00, 'Electronic check', 'Fiber optic', 'No')
ON CONFLICT (customer_id) DO NOTHING;

-- Insert sample usage logs
INSERT INTO usage_logs (customer_id, date, data_usage_gb, call_minutes, sms_count) VALUES
('CUST_001', CURRENT_DATE - INTERVAL '1 day', 15.5, 120, 25),
('CUST_001', CURRENT_DATE - INTERVAL '2 days', 18.2, 95, 30),
('CUST_002', CURRENT_DATE - INTERVAL '1 day', 8.7, 200, 15),
('CUST_002', CURRENT_DATE - INTERVAL '2 days', 12.1, 180, 20),
('CUST_003', CURRENT_DATE - INTERVAL '1 day', 22.3, 45, 40),
('CUST_003', CURRENT_DATE - INTERVAL '2 days', 25.8, 60, 35);

-- Insert sample support tickets
INSERT INTO support_tickets (customer_id, category, priority, status, description) VALUES
('CUST_001', 'Technical', 'Medium', 'Closed', 'Internet connection issues resolved'),
('CUST_002', 'Billing', 'Low', 'Open', 'Question about monthly charges'),
('CUST_003', 'Technical', 'High', 'In Progress', 'Frequent service outages');

-- Create MLflow database
CREATE DATABASE mlflow_db;
