#!/bin/bash
set -e

# Configuration
REGISTRY="your-ecr-registry"  # Replace with your AWS ECR registry
IMAGE_NAME="copilot"
TAG="latest"
AWS_REGION="us-east-1"
ECS_CLUSTER="copilot-cluster"
ECS_SERVICE="copilot-service"

# Build and push Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .
docker tag $IMAGE_NAME:$TAG $REGISTRY/$IMAGE_NAME:$TAG

echo "Logging into AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

echo "Pushing Docker image to ECR..."
docker push $REGISTRY/$IMAGE_NAME:$TAG

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --force-new-deployment --region $AWS_REGION

# Verify monitoring services
echo "Checking Prometheus and Grafana..."
curl -s http://localhost:9090/-/healthy || echo "Prometheus not running"
curl -s http://localhost:3000/api/health || echo "Grafana not running"

echo "Deployment completed successfully!"