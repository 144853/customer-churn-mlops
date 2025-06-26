# Model Deployment Guide

This guide covers the complete process of deploying machine learning models in the Customer Churn Prediction MLOps system.

## 🎯 **Overview**

Our model deployment process follows MLOps best practices with automated validation, staged rollouts, and comprehensive monitoring.

## 📋 **Prerequisites**

### **Development Environment**
- Python 3.9+
- Docker and Docker Compose
- Access to MLflow tracking server
- Kubernetes cluster access (for production)
- AWS credentials configured

### **Model Requirements**
- **Performance**: ROC-AUC ≥ 0.85
- **Latency**: Inference time < 300ms
- **Stability**: Consistent performance across CV folds
- **Documentation**: Complete model card
- **Validation**: Passed all quality gates

## 🚀 **Deployment Pipeline**

### **Stage 1: Model Training and Validation**

```bash
# 1. Train new model
python scripts/train_model.py --trigger manual

# 2. Validate model performance
python scripts/model_management.py validate --model-path models/trained_models/churn_model_v1.2.0.joblib

# 3. Generate model card
python scripts/generate_model_card.py \
  --evaluation-results models/model_metadata/performance_reports/eval_v1.2.0.json \
  --output models/model_metadata/model_cards/model_card_v1.2.0.md \
  --model-version 1.2.0 \
  --model-type "XGBoost Classifier"
```

### **Stage 2: Staging Deployment**

```bash
# 1. Deploy to staging environment
kubectl apply -f k8s/staging/

# 2. Update staging deployment with new model
kubectl set image deployment/churn-prediction-staging \
  churn-prediction=customer-churn-mlops:v1.2.0 \
  -n mlops-staging

# 3. Run integration tests
pytest tests/integration/ --env staging

# 4. Performance testing
python scripts/load_test.py --endpoint https://staging-api.company.com --duration 300
```

### **Stage 3: Production Deployment**

```bash
# 1. Promote model to production registry
python scripts/model_management.py promote \
  --version 1.2.0 \
  --model-path models/trained_models/churn_model_v1.2.0.joblib

# 2. Deploy to production using blue-green strategy
./scripts/deploy.sh production --strategy blue-green

# 3. Verify deployment
kubectl rollout status deployment/churn-prediction -n mlops-production

# 4. Run smoke tests
curl -X GET https://api.company.com/health
curl -X POST https://api.company.com/predict -d @test_data.json
```

## 🔄 **Deployment Strategies**

### **Blue-Green Deployment**

```yaml
# Blue environment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction-blue
  labels:
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
      version: blue
  template:
    spec:
      containers:
      - name: churn-prediction
        image: customer-churn-mlops:v1.1.0

---
# Green environment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-prediction-green
  labels:
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-prediction
      version: green
  template:
    spec:
      containers:
      - name: churn-prediction
        image: customer-churn-mlops:v1.2.0
```

### **Canary Deployment**

```bash
# 1. Deploy canary with 10% traffic
kubectl apply -f k8s/canary/
istioctl create -f istio/canary-10percent.yaml

# 2. Monitor canary metrics
kubectl logs -f deployment/churn-prediction-canary

# 3. Gradually increase traffic
istioctl replace -f istio/canary-50percent.yaml
istioctl replace -f istio/canary-100percent.yaml

# 4. Complete rollout
kubectl delete -f k8s/canary/
```

## 📏 **Model Validation Checklist**

### **Performance Validation**
- [ ] ROC-AUC ≥ 0.85 on test set
- [ ] Precision ≥ 0.75
- [ ] Recall ≥ 0.70
- [ ] F1-Score ≥ 0.72
- [ ] Cross-validation stability (std < 0.05)

### **Technical Validation**
- [ ] Inference latency < 300ms
- [ ] Memory usage < 2GB
- [ ] Model size < 500MB
- [ ] Compatible with serving infrastructure
- [ ] Input/output schema validation

### **Business Validation**
- [ ] Performance on key customer segments
- [ ] Bias and fairness testing passed
- [ ] Business impact simulation positive
- [ ] Stakeholder approval obtained

### **Operational Validation**
- [ ] Model card completed
- [ ] Monitoring setup configured
- [ ] Rollback plan documented
- [ ] Alert thresholds defined

## 🛡️ **Rollback Procedures**

### **Automatic Rollback Triggers**
- ROC-AUC drops below 0.80
- Latency exceeds 500ms
- Error rate > 5%
- Data drift score > 0.2

### **Manual Rollback**

```bash
# 1. Emergency rollback to previous version
python scripts/model_management.py rollback --target previous

# 2. Update Kubernetes deployment
kubectl rollout undo deployment/churn-prediction -n mlops-production

# 3. Verify rollback
kubectl get pods -n mlops-production
curl -X GET https://api.company.com/model/info

# 4. Update monitoring alerts
python scripts/update_alerts.py --model-version 1.1.0
```

### **Rollback to Stable Version**

```bash
# Rollback to last known good model
python scripts/model_management.py rollback --target rollback

# Verify model version
curl -X GET https://api.company.com/model/info | jq '.model_version'
```

## 📊 **Monitoring and Observability**

### **Key Metrics to Monitor**

```python
# Performance metrics
model_performance_gauge.labels(metric_type="roc_auc").set(0.87)
model_performance_gauge.labels(metric_type="precision").set(0.82)
model_performance_gauge.labels(metric_type="recall").set(0.79)

# Operational metrics
prediction_latency_histogram.observe(0.045)  # 45ms
prediction_counter.labels(risk_level="high").inc()
error_counter.labels(error_type="timeout").inc()

# Business metrics
churn_prevention_gauge.set(0.15)  # 15% churn reduction
campaign_efficiency_gauge.set(3.2)  # 3.2x improvement
```

### **Alerting Rules**

```yaml
# Prometheus alerting rules
groups:
  - name: model_performance
    rules:
      - alert: ModelPerformanceDegradation
        expr: model_performance_score{metric_type="roc_auc"} < 0.80
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model performance has degraded"
          description: "ROC-AUC has dropped to {{ $value }}"

      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, prediction_latency_seconds_bucket) > 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
```

## 🔒 **Security Considerations**

### **Model Security**
- Models are cryptographically signed
- Access controlled via Kubernetes RBAC
- Network policies restrict traffic
- Secrets managed via AWS Secrets Manager

### **Data Security**
- Input data validation and sanitization
- PII detection and masking
- Audit logging for all predictions
- Encryption in transit and at rest

### **Compliance**
- PIPEDA/GDPR compliance checks
- Model explainability requirements
- Data retention policies
- Regular security audits

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Model Loading Failures**
```bash
# Check model file integrity
ls -la models/production/current_model.joblib
python -c "import joblib; print(joblib.load('models/production/current_model.joblib'))"

# Verify model registry
python scripts/model_management.py status

# Check MLflow connectivity
curl -f http://localhost:5000/health
```

#### **High Latency Issues**
```bash
# Check resource usage
kubectl top pods -n mlops-production

# Review model complexity
python scripts/model_profiler.py --model-path models/production/current_model.joblib

# Optimize model if needed
python scripts/model_optimizer.py --input-model current_model.joblib --output optimized_model.joblib
```

#### **Performance Degradation**
```bash
# Check for data drift
python scripts/drift_analysis.py --reference-data data/reference.parquet --current-data data/latest.parquet

# Analyze prediction distribution
python scripts/prediction_analysis.py --date-range "2025-06-01,2025-06-26"

# Compare with baseline
python scripts/model_comparison.py --model1 current --model2 baseline
```

### **Emergency Procedures**

#### **Complete Service Failure**
```bash
# 1. Immediate rollback
python scripts/model_management.py rollback --target rollback
kubectl rollout undo deployment/churn-prediction -n mlops-production

# 2. Scale up replicas
kubectl scale deployment churn-prediction --replicas=10 -n mlops-production

# 3. Enable fallback service
kubectl apply -f k8s/fallback-service.yaml

# 4. Notify stakeholders
python scripts/send_alert.py --type "service_failure" --severity "critical"
```

## 📚 **Best Practices**

### **Pre-Deployment**
1. **Comprehensive Testing**: Unit, integration, and load testing
2. **Performance Validation**: Verify all SLA requirements
3. **Security Review**: Complete security checklist
4. **Documentation**: Update all relevant documentation
5. **Stakeholder Approval**: Get business and technical sign-off

### **During Deployment**
1. **Gradual Rollout**: Use canary or blue-green deployment
2. **Real-time Monitoring**: Watch all key metrics closely
3. **Quick Rollback**: Be ready to rollback immediately
4. **Communication**: Keep stakeholders informed
5. **Documentation**: Log all deployment steps

### **Post-Deployment**
1. **Performance Monitoring**: Track metrics for 24-48 hours
2. **User Feedback**: Collect feedback from business users
3. **Issue Tracking**: Monitor for any issues or anomalies
4. **Documentation**: Update deployment logs and lessons learned
5. **Retrospective**: Conduct post-deployment review

## 📞 **Support Contacts**

- **MLOps Team**: mlops@telecomcorp.com
- **On-Call Engineer**: +1-555-MLOPS-1 (24/7)
- **Escalation**: VP Engineering
- **Business Contact**: Product Owner
- **Emergency Hotline**: +1-555-EMERGENCY

---

**Remember**: Always test in staging before production deployment and have a rollback plan ready.
