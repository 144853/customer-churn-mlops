#!/bin/bash

# Customer Churn MLOps Deployment Script
# This script handles the deployment of the customer churn prediction service

set -e  # Exit on any error

# Configuration
APP_NAME="customer-churn-mlops"
DOCKER_IMAGE="$APP_NAME:latest"
KUBES_NAMESPACE="mlops"
HELM_RELEASE_NAME="churn-prediction"
MONITORING_NAMESPACE="monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required tools are installed
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v helm &> /dev/null; then
        missing_tools+=("helm")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi
    
    log_success "All prerequisites are installed"
}

# Function to build Docker image
build_docker_image() {
    log_info "Building Docker image: $DOCKER_IMAGE"
    
    # Build the main application image
    docker build -t $DOCKER_IMAGE .
    
    # Tag with git commit hash if available
    if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
        GIT_HASH=$(git rev-parse --short HEAD)
        docker tag $DOCKER_IMAGE "$APP_NAME:$GIT_HASH"
        log_info "Tagged image with git hash: $APP_NAME:$GIT_HASH"
    fi
    
    log_success "Docker image built successfully"
}

# Function to create Kubernetes namespace
create_namespace() {
    log_info "Creating Kubernetes namespace: $KUBES_NAMESPACE"
    
    if kubectl get namespace $KUBES_NAMESPACE &> /dev/null; then
        log_warning "Namespace $KUBES_NAMESPACE already exists"
    else
        kubectl create namespace $KUBES_NAMESPACE
        log_success "Namespace $KUBES_NAMESPACE created"
    fi
}

# Function to deploy with Helm
deploy_with_helm() {
    log_info "Deploying application with Helm"
    
    # Check if Helm chart exists
    if [ ! -d "helm/churn-prediction" ]; then
        log_error "Helm chart not found at helm/churn-prediction"
        exit 1
    fi
    
    # Deploy or upgrade the Helm release
    helm upgrade --install $HELM_RELEASE_NAME ./helm/churn-prediction \
        --namespace $KUBES_NAMESPACE \
        --set image.repository=$APP_NAME \
        --set image.tag=latest \
        --wait --timeout=300s
    
    log_success "Application deployed successfully with Helm"
}

# Function to deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack"
    
    # Create monitoring namespace
    if kubectl get namespace $MONITORING_NAMESPACE &> /dev/null; then
        log_warning "Namespace $MONITORING_NAMESPACE already exists"
    else
        kubectl create namespace $MONITORING_NAMESPACE
        log_success "Namespace $MONITORING_NAMESPACE created"
    fi
    
    # Deploy Prometheus
    if ! helm list -n $MONITORING_NAMESPACE | grep -q prometheus; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace $MONITORING_NAMESPACE \
            --set grafana.adminPassword=admin123 \
            --wait
        
        log_success "Prometheus and Grafana deployed"
    else
        log_warning "Prometheus already deployed"
    fi
}

# Function to run health checks
run_health_checks() {
    log_info "Running health checks"
    
    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=churn-prediction -n $KUBES_NAMESPACE --timeout=300s
    
    # Get service URL
    SERVICE_URL=$(kubectl get svc -n $KUBES_NAMESPACE churn-prediction -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_URL" ]; then
        # If LoadBalancer IP is not available, use port-forward for testing
        log_warning "LoadBalancer IP not available, using port-forward for health check"
        kubectl port-forward -n $KUBES_NAMESPACE svc/churn-prediction 8080:8000 &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_URL="localhost:8080"
    fi
    
    # Health check
    log_info "Checking health endpoint: http://$SERVICE_URL/health"
    
    for i in {1..10}; do
        if curl -f -s "http://$SERVICE_URL/health" > /dev/null; then
            log_success "Health check passed"
            break
        else
            log_warning "Health check attempt $i/10 failed, retrying in 10 seconds..."
            sleep 10
        fi
        
        if [ $i -eq 10 ]; then
            log_error "Health check failed after 10 attempts"
            if [ ! -z "$PORT_FORWARD_PID" ]; then
                kill $PORT_FORWARD_PID
            fi
            exit 1
        fi
    done
    
    # Clean up port-forward if used
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID
    fi
}

# Function to show deployment info
show_deployment_info() {
    log_info "Deployment Information"
    echo "========================================"
    
    # Show pods
    echo "Pods in $KUBES_NAMESPACE namespace:"
    kubectl get pods -n $KUBES_NAMESPACE
    echo ""
    
    # Show services
    echo "Services in $KUBES_NAMESPACE namespace:"
    kubectl get svc -n $KUBES_NAMESPACE
    echo ""
    
    # Show Helm releases
    echo "Helm releases:"
    helm list -n $KUBES_NAMESPACE
    echo ""
    
    # Show access information
    echo "Access Information:"
    echo "- API Documentation: http://<service-ip>:8000/docs"
    echo "- Health Check: http://<service-ip>:8000/health"
    echo "- Metrics: http://<service-ip>:8000/metrics"
    echo ""
    
    # Grafana access info
    echo "Monitoring:"
    echo "- Grafana: http://<grafana-ip>:3000 (admin/admin123)"
    echo "- Prometheus: http://<prometheus-ip>:9090"
    echo ""
    
    echo "To get service IPs, run:"
    echo "kubectl get svc -n $KUBES_NAMESPACE"
    echo "kubectl get svc -n $MONITORING_NAMESPACE"
}

# Function to cleanup deployment
cleanup_deployment() {
    log_info "Cleaning up deployment"
    
    # Delete Helm release
    if helm list -n $KUBES_NAMESPACE | grep -q $HELM_RELEASE_NAME; then
        helm uninstall $HELM_RELEASE_NAME -n $KUBES_NAMESPACE
        log_success "Helm release deleted"
    fi
    
    # Delete namespace
    if kubectl get namespace $KUBES_NAMESPACE &> /dev/null; then
        kubectl delete namespace $KUBES_NAMESPACE
        log_success "Namespace deleted"
    fi
    
    # Remove Docker images
    if docker images | grep -q $APP_NAME; then
        docker rmi $(docker images $APP_NAME -q) --force
        log_success "Docker images removed"
    fi
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            log_info "Starting deployment process"
            check_prerequisites
            build_docker_image
            create_namespace
            deploy_with_helm
            deploy_monitoring
            run_health_checks
            show_deployment_info
            log_success "Deployment completed successfully!"
            ;;
        "build")
            log_info "Building Docker image only"
            check_prerequisites
            build_docker_image
            ;;
        "health")
            log_info "Running health checks only"
            run_health_checks
            ;;
        "info")
            show_deployment_info
            ;;
        "cleanup")
            log_warning "This will delete the entire deployment. Are you sure? (y/N)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                cleanup_deployment
            else
                log_info "Cleanup cancelled"
            fi
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    - Full deployment (default)"
            echo "  build     - Build Docker image only"
            echo "  health    - Run health checks"
            echo "  info      - Show deployment information"
            echo "  cleanup   - Remove entire deployment"
            echo "  help      - Show this help message"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
