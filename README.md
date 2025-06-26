# Customer Churn Prediction MLOps

A comprehensive MLOps solution for predicting customer churn in telecommunications, featuring end-to-end automation, monitoring, and production deployment capabilities.

![MLOps Pipeline](https://img.shields.io/badge/MLOps-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange)

## 🎯 Project Overview

This project implements a production-ready machine learning system for predicting customer churn with the following key features:

- **End-to-end ML Pipeline**: From data ingestion to model serving
- **Real-time Predictions**: FastAPI-based REST API with <300ms latency
- **Automated Retraining**: Weekly scheduled retraining with drift detection
- **Comprehensive Monitoring**: Prometheus metrics, Grafana dashboards, and alerting
- **Production Deployment**: Kubernetes with Helm charts and CI/CD pipelines
- **Data Quality**: Great Expectations for data validation
- **Model Versioning**: MLflow for experiment tracking and model registry

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[Customer Profiles - PostgreSQL]
        B[Usage Logs - S3]
        C[Billing Data - CSV]
        D[Support Tickets - API]
    end
    
    subgraph "Data Pipeline"
        E[Data Ingestion - Airflow]
        F[Data Validation - Great Expectations]
        G[Feature Engineering]
    end
    
    subgraph "ML Pipeline"
        H[Model Training - MLflow]
        I[Hyperparameter Tuning - Optuna]
        J[Model Validation]
        K[Model Registry]
    end
    
    subgraph "Serving"
        L[FastAPI Service]
        M[Load Balancer]
        N[Kubernetes Cluster]
    end
    
    subgraph "Monitoring"
        O[Prometheus]
        P[Grafana]
        Q[Drift Detection - Evidently]
        R[Alerting]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    L --> O
    O --> P
    O --> R
    G --> Q
    Q --> H
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes cluster (for production deployment)
- AWS account (for S3 and other services)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/144853/customer-churn-mlops.git
   cd customer-churn-mlops
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start local services**
   ```bash
   docker-compose up -d
   ```

5. **Run the API**
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```

6. **Access the API documentation**
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Production Deployment

1. **Deploy to Kubernetes**
   ```bash
   ./scripts/deploy.sh
   ```

2. **Monitor the deployment**
   ```bash
   ./scripts/deploy.sh info
   ```

## 📊 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_data": {
         "customer_id": "CUST_12345",
         "age": 35,
         "gender": "F",
         "tenure_months": 24,
         "contract_type": "Two year",
         "monthly_charges": 79.99,
         "total_charges": 1919.76,
         "payment_method": "Credit card",
         "internet_service": "Fiber optic",
         "phone_service": "Yes"
       }
     }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "customers_data": [
         {
           "customer_id": "CUST_12345",
           "age": 35,
           "monthly_charges": 79.99
         },
         {
           "customer_id": "CUST_12346",
           "age": 45,
           "monthly_charges": 65.50
         }
       ]
     }'
```

### Using the CLI

```bash
# Single prediction
python scripts/run_inference.py --input '{"customer_id": "CUST_001", "age": 35}'

# Batch prediction from CSV
python scripts/run_inference.py --input customers.csv --batch --output predictions.json
```

## 🔧 Training Pipeline

### Manual Training

```bash
# Run full training pipeline
python scripts/train_model.py

# Check if retraining is needed
python scripts/train_model.py --check-only

# Force retraining
python scripts/train_model.py --trigger=manual
```

### Automated Retraining

The system automatically checks for data drift daily and triggers retraining when:
- Data drift is detected (>10% drift score)
- Model performance degrades
- Weekly scheduled retraining

## 📈 Monitoring & Observability

### Metrics Tracked

- **Performance Metrics**: Latency, throughput, error rates
- **Model Metrics**: Prediction confidence, ROC-AUC, accuracy
- **Data Quality**: Drift detection, data validation results
- **Infrastructure**: CPU, memory, disk usage

### Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **MLflow UI**: http://localhost:5000

### Alerts

Configured alerts for:
- High prediction latency (>300ms)
- Model performance degradation
- Data drift detection
- Service availability issues
- Resource utilization

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_api.py
pytest tests/test_models.py
pytest tests/test_data.py
```

## 🔄 CI/CD Pipeline

The project includes comprehensive GitHub Actions workflows:

### Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)
- Code quality checks (black, isort, flake8, mypy)
- Unit and integration tests
- Docker image building and pushing
- Automated deployment to staging
- Security scanning
- Production deployment with approval

### Retraining Pipeline (`.github/workflows/retraining-check.yml`)
- Daily drift detection
- Automated model retraining
- Production model updates
- Notification and reporting

## 📁 Project Structure

```
customer-churn-mlops/
├── src/
│   ├── api/                    # FastAPI application
│   ├── config/                 # Configuration management
│   ├── data/                   # Data ingestion and validation
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and prediction
│   ├── monitoring/             # Drift detection and monitoring
│   └── training/               # Training pipeline orchestration
├── tests/                      # Unit and integration tests
├── scripts/                    # Utility scripts
├── helm/                       # Kubernetes Helm charts
├── monitoring/                 # Monitoring configurations
├── .github/workflows/          # CI/CD pipelines
├── docker-compose.yml          # Local development setup
├── Dockerfile                  # Container definition
└── requirements.txt            # Python dependencies
```

## 🔒 Security & Compliance

- **Data Privacy**: PIPEDA and GDPR compliance
- **Access Control**: Role-based access control (RBAC)
- **Secrets Management**: AWS Secrets Manager integration
- **Container Security**: Trivy vulnerability scanning
- **Audit Trail**: Complete model lifecycle logging

## 📖 Documentation

### API Documentation
- Interactive API docs available at `/docs` endpoint
- OpenAPI specification included

### Model Documentation
- Model cards for each trained model
- Performance reports and test results
- Feature importance analysis

### Architecture Documentation
- System architecture diagrams
- Data flow documentation
- Deployment guides

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation for new features
- Use conventional commit messages

## 📋 Performance Requirements

- **Latency**: <300ms for real-time predictions
- **ROC-AUC**: ≥85% on test data
- **Availability**: 99.9% uptime
- **Throughput**: 1000+ predictions per second
- **Data Processing**: ~5M records updated daily

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/churn_db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your-bucket

# Model
MODEL_THRESHOLD=0.5
MAX_LATENCY_MS=300
MIN_ROC_AUC=0.85

# Monitoring
PROMETHEUS_PORT=8001
LOG_LEVEL=INFO
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Issues**
   ```bash
   # Check model files
   ls -la models/
   # Verify MLflow connection
   curl http://localhost:5000/health
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connection
   docker-compose exec postgres psql -U postgres -d churn_db -c "SELECT 1;"
   ```

3. **High Latency**
   - Check resource allocation
   - Review feature engineering pipeline
   - Optimize model complexity

4. **Drift Detection**
   - Verify reference data exists
   - Check data quality
   - Review drift thresholds

### Logs and Debugging

```bash
# Application logs
docker-compose logs app

# Kubernetes logs
kubectl logs -f deployment/churn-prediction -n mlops

# Check metrics
curl http://localhost:8000/metrics
```

## 📞 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the documentation at `/docs`
- Review the troubleshooting guide above

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TelecomCorp Inc. for the business requirements
- Open source community for the amazing tools and libraries
- MLOps community for best practices and patterns

---

**Built with ❤️ for production ML systems**
