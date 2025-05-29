# Production Deployment Guide for LLM Safety System

## Overview

This guide provides comprehensive instructions for deploying the LLM Safety system in production environments. The system includes red teaming capabilities, safety classification, and multiple mitigation techniques.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Deployment Options](#deployment-options)
4. [Configuration](#configuration)
5. [Security Considerations](#security-considerations)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Scaling Strategies](#scaling-strategies)
8. [Maintenance and Updates](#maintenance-and-updates)
9. [Troubleshooting](#troubleshooting)

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Safety Filter   │───▶│   LLM Backend   │
│                 │    │   Classifier     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Mitigation     │
                       │   Techniques     │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Response Post-  │
                       │   Processing     │
                       └──────────────────┘
```

### Core Components

1. **Safety Classifier**: DistilBERT-based three-class classifier
2. **Mitigation Engine**: Multiple techniques (rejection sampling, CoT moderation, prompt updating)
3. **Red Teaming Module**: Adversarial prompt generation and testing
4. **Monitoring System**: Real-time safety metrics and alerting
5. **API Gateway**: RESTful API for integration

## Infrastructure Requirements

### Minimum Requirements

**Development/Testing Environment:**

- CPU: 4 cores, 2.4GHz
- RAM: 16GB
- Storage: 50GB SSD
- GPU: Optional (NVIDIA GTX 1060 or equivalent)

**Production Environment:**

- CPU: 8+ cores, 2.8GHz
- RAM: 32GB+
- Storage: 200GB+ SSD
- GPU: NVIDIA V100/A100 or equivalent (recommended)
- Network: High-bandwidth, low-latency

### Software Dependencies

```bash
# Core Python environment
Python 3.8+
PyTorch 1.9+
Transformers 4.15+
FastAPI 0.75+
uvicorn 0.17+

# Additional dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
requests>=2.27.0
```

### Docker Requirements

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Deployment Options

### Option 1: Cloud Platform Deployment (Recommended)

#### AWS Deployment

**1. Container Deployment (ECS/EKS)**

```yaml
# docker-compose.yml
version: "3.8"
services:
  safety-classifier:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/safety_classifier.pth
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - safety-classifier
```

**2. Lambda Deployment (Serverless)**

```python
# lambda_handler.py
import json
import boto3
from src.safety_filter_classifier.classifier import SafetyClassifier

# Initialize model (cold start optimization)
classifier = None

def lambda_handler(event, context):
    global classifier

    if classifier is None:
        # Load model from S3
        classifier = SafetyClassifier.load_model("s3://your-bucket/models/safety_classifier.pth")

    # Extract input
    text = event.get('text', '')

    # Classify
    result = classifier.predict(text)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'safety_label': result[0],
            'confidence': result[1],
            'processing_time': result[2] if len(result) > 2 else None
        })
    }
```

#### Google Cloud Platform (GCP)

**Cloud Run Deployment:**

```yaml
# cloudbuild.yaml
steps:
  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/llm-safety:$SHORT_SHA", "."]
  - name: "gcr.io/cloud-builders/docker"
    args: ["push", "gcr.io/$PROJECT_ID/llm-safety:$SHORT_SHA"]
  - name: "gcr.io/cloud-builders/gcloud"
    args:
      [
        "run",
        "deploy",
        "llm-safety",
        "--image",
        "gcr.io/$PROJECT_ID/llm-safety:$SHORT_SHA",
        "--platform",
        "managed",
        "--region",
        "us-central1",
        "--memory",
        "4Gi",
        "--cpu",
        "2",
        "--max-instances",
        "10",
      ]
```

#### Azure Deployment

**Container Instances:**

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group llm-safety-rg \
  --name llm-safety-api \
  --image youracr.azurecr.io/llm-safety:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    MODEL_PATH=/app/models/safety_classifier.pth \
    LOG_LEVEL=INFO
```

### Option 2: On-Premises Deployment

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-safety-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-safety
  template:
    metadata:
      labels:
        app: llm-safety
    spec:
      containers:
        - name: llm-safety
          image: llm-safety:latest
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: "/app/models/safety_classifier.pth"
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: llm-safety-service
spec:
  selector:
    app: llm-safety
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### Docker Swarm Deployment

```yaml
# docker-stack.yml
version: "3.8"
services:
  llm-safety:
    image: llm-safety:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/safety_classifier.pth
    volumes:
      - models:/app/models
      - logs:/app/logs
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

volumes:
  models:
  logs:
```

## Configuration

### Environment Variables

```bash
# Core Configuration
MODEL_PATH=/app/models/safety_classifier.pth
API_PORT=8000
LOG_LEVEL=INFO
WORKERS=4

# Database Configuration (if applicable)
DATABASE_URL=postgresql://user:pass@localhost:5432/llm_safety
REDIS_URL=redis://localhost:6379

# Security Configuration
API_KEY_REQUIRED=true
JWT_SECRET=your-secret-key
CORS_ORIGINS=["https://yourdomain.com"]

# Model Configuration
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=16
TEMPERATURE=0.7

# Mitigation Configuration
ENABLE_REJECTION_SAMPLING=true
ENABLE_COT_MODERATION=true
ENABLE_PROMPT_UPDATING=true
MAX_REJECTION_ATTEMPTS=5

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=your-sentry-dsn

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
```

### Configuration File (config.yaml)

```yaml
# config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false

model:
  safety_classifier_path: "models/safety_classifier.pth"
  max_sequence_length: 512
  batch_size: 16
  device: "auto" # "cpu", "cuda", or "auto"

mitigation:
  rejection_sampling:
    enabled: true
    max_attempts: 5
    temperature: 0.3

  chain_of_thought:
    enabled: true
    temperature: 0.2

  prompt_updating:
    enabled: true
    risk_threshold: 0.7

security:
  api_key_required: true
  cors_origins:
    - "https://yourdomain.com"
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    requests_per_hour: 1000

monitoring:
  metrics_enabled: true
  log_level: "INFO"
  sentry_dsn: null

database:
  url: null # Optional: for storing annotations/feedback

cache:
  redis_url: "redis://localhost:6379"
  ttl: 3600 # Cache TTL in seconds
```

## Security Considerations

### 1. Authentication and Authorization

```python
# api/auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Usage in API endpoints
@app.post("/classify", dependencies=[Depends(verify_token)])
async def classify_text(request: ClassificationRequest):
    # Implementation
    pass
```

### 2. Input Validation and Sanitization

```python
# api/validation.py
from pydantic import BaseModel, validator
from typing import Optional

class ClassificationRequest(BaseModel):
    text: str
    risk_level: Optional[str] = "medium"

    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:  # Limit input size
            raise ValueError('Text too long')
        return v.strip()

    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError('Invalid risk level')
        return v
```

### 3. Rate Limiting

```python
# api/rate_limiting.py
from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.state.limiter = limiter
@app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/classify")
@limiter.limit("100/minute")
async def classify_text(request: Request, data: ClassificationRequest):
    # Implementation
    pass
```

### 4. HTTPS and SSL Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://safety-classifier:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring and Logging

### 1. Application Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Metrics
classification_requests = Counter('safety_classifications_total', 'Total safety classifications')
classification_latency = Histogram('safety_classification_duration_seconds', 'Classification latency')
active_connections = Gauge('active_connections', 'Active connections')
unsafe_content_detected = Counter('unsafe_content_total', 'Unsafe content detected', ['safety_label'])

def track_classification(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        classification_requests.inc()

        try:
            result = func(*args, **kwargs)

            # Track safety label
            if result and len(result) > 0:
                unsafe_content_detected.labels(safety_label=result[0]).inc()

            return result
        finally:
            classification_latency.observe(time.time() - start_time)

    return wrapper

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            if hasattr(record, 'safety_label'):
                log_data['safety_label'] = record.safety_label
            if hasattr(record, 'confidence'):
                log_data['confidence'] = record.confidence
            if hasattr(record, 'processing_time'):
                log_data['processing_time'] = record.processing_time

            return json.dumps(log_data)

# Usage
logger = StructuredLogger(__name__)

def classify_with_logging(text: str):
    result = classifier.predict(text)

    logger.logger.info(
        "Safety classification completed",
        extra={
            'safety_label': result[0],
            'confidence': result[1],
            'processing_time': result[2]
        }
    )

    return result
```

### 3. Health Checks

```python
# api/health.py
from fastapi import HTTPException
import psutil
import torch

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }

    # Model availability
    try:
        if hasattr(app.state, 'classifier'):
            health_status["checks"]["model"] = "loaded"
        else:
            health_status["checks"]["model"] = "not_loaded"
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["checks"]["model"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"

    # Memory usage
    memory = psutil.virtual_memory()
    health_status["checks"]["memory_usage"] = f"{memory.percent}%"

    if memory.percent > 90:
        health_status["status"] = "degraded"

    # GPU availability (if applicable)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
        health_status["checks"]["gpu_memory"] = f"{gpu_memory:.1%}"

    if health_status["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status

@app.get("/ready")
async def readiness_check():
    # Check if model is loaded and ready
    if not hasattr(app.state, 'classifier'):
        raise HTTPException(status_code=503, detail="Model not ready")

    return {"status": "ready"}
```

## Scaling Strategies

### 1. Horizontal Scaling

```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-safety-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-safety-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### 2. Load Balancing

```python
# load_balancer.py
from fastapi import FastAPI
import httpx
import asyncio
from typing import List

class LoadBalancer:
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.current = 0

    def next_backend(self) -> str:
        backend = self.backends[self.current]
        self.current = (self.current + 1) % len(self.backends)
        return backend

    async def classify(self, text: str):
        backend = self.next_backend()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend}/classify",
                json={"text": text}
            )
            return response.json()

# Usage
load_balancer = LoadBalancer([
    "http://safety-1:8000",
    "http://safety-2:8000",
    "http://safety-3:8000"
])
```

### 3. Caching Strategy

```python
# caching.py
import redis
import json
import hashlib
from typing import Optional

class SafetyCache:
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl

    def _get_key(self, text: str) -> str:
        return f"safety:{hashlib.sha256(text.encode()).hexdigest()}"

    def get(self, text: str) -> Optional[dict]:
        key = self._get_key(text)
        cached = self.redis_client.get(key)

        if cached:
            return json.loads(cached)
        return None

    def set(self, text: str, result: dict):
        key = self._get_key(text)
        self.redis_client.setex(
            key,
            self.ttl,
            json.dumps(result)
        )

# Usage in API
cache = SafetyCache("redis://localhost:6379")

@app.post("/classify")
async def classify_with_cache(request: ClassificationRequest):
    # Check cache first
    cached_result = cache.get(request.text)
    if cached_result:
        return cached_result

    # Classify
    result = classifier.predict(request.text)

    # Cache result
    result_dict = {
        "safety_label": result[0],
        "confidence": result[1]
    }
    cache.set(request.text, result_dict)

    return result_dict
```

## Maintenance and Updates

### 1. Model Updates

```python
# model_updater.py
import os
import shutil
from pathlib import Path

class ModelUpdater:
    def __init__(self, model_dir: str, backup_dir: str):
        self.model_dir = Path(model_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

    def backup_current_model(self):
        if self.model_dir.exists():
            backup_path = self.backup_dir / f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(self.model_dir, backup_path)
            return backup_path
        return None

    def update_model(self, new_model_path: str):
        # Backup current model
        backup_path = self.backup_current_model()

        try:
            # Update model
            shutil.copytree(new_model_path, self.model_dir)

            # Test new model
            if self.test_model():
                print("Model update successful")
                return True
            else:
                # Rollback
                if backup_path:
                    shutil.rmtree(self.model_dir)
                    shutil.copytree(backup_path, self.model_dir)
                print("Model update failed, rolled back")
                return False

        except Exception as e:
            print(f"Error updating model: {e}")
            return False

    def test_model(self) -> bool:
        try:
            from src.safety_filter_classifier.classifier import SafetyClassifier
            classifier = SafetyClassifier.load_model(str(self.model_dir / "safety_classifier.pth"))

            # Test with sample input
            result = classifier.predict("This is a test")
            return len(result) >= 2  # Should return label and confidence

        except Exception as e:
            print(f"Model test failed: {e}")
            return False
```

### 2. Automated Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ["v*"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: |
          docker build -t llm-safety:${{ github.sha }} .
          docker tag llm-safety:${{ github.sha }} llm-safety:latest

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push llm-safety:${{ github.sha }}
          docker push llm-safety:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Update Kubernetes deployment
          kubectl set image deployment/llm-safety-deployment llm-safety=llm-safety:${{ github.sha }}
          kubectl rollout status deployment/llm-safety-deployment
```

### 3. Database Migrations

```python
# migrations/migrate.py
import sqlite3
from pathlib import Path

class DatabaseMigrator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations_dir = Path("migrations")

    def get_current_version(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
            result = cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def apply_migrations(self):
        current_version = self.get_current_version()
        migration_files = sorted(self.migrations_dir.glob("*.sql"))

        for migration_file in migration_files:
            version = int(migration_file.stem.split("_")[0])

            if version > current_version:
                self.apply_migration(migration_file, version)

    def apply_migration(self, migration_file: Path, version: int):
        conn = sqlite3.connect(self.db_path)
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()

            conn.executescript(migration_sql)

            # Record migration
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
                (version,)
            )

            conn.commit()
            print(f"Applied migration {migration_file.name}")

        except Exception as e:
            conn.rollback()
            print(f"Failed to apply migration {migration_file.name}: {e}")
            raise
        finally:
            conn.close()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

**Issue**: Model consuming too much memory
**Solution**:

```python
# memory_optimization.py
import torch
import gc

def optimize_memory():
    # Clear cache
    torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    # Use mixed precision
    torch.backends.cudnn.benchmark = True

# Model loading with memory optimization
def load_model_optimized(model_path: str):
    # Load on CPU first
    model = SafetyClassifier.load_model(model_path, map_location='cpu')

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        model.half()  # Use half precision

    return model
```

#### 2. Slow Response Times

**Issue**: Classification taking too long
**Solution**:

```python
# performance_optimization.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedClassifier:
    def __init__(self, model_path: str, max_workers: int = 4):
        self.classifier = SafetyClassifier.load_model(model_path)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def classify_async(self, text: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.classifier.predict,
            text
        )

    def classify_batch(self, texts: List[str]):
        return self.classifier.batch_predict(texts)
```

#### 3. Model Loading Failures

**Issue**: Model fails to load
**Solution**:

```python
# model_diagnostics.py
def diagnose_model_loading(model_path: str):
    checks = {
        'file_exists': Path(model_path).exists(),
        'file_readable': os.access(model_path, os.R_OK),
        'file_size': Path(model_path).stat().st_size if Path(model_path).exists() else 0,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }

    # Try loading
    try:
        model = torch.load(model_path, map_location='cpu')
        checks['model_loadable'] = True
        checks['model_keys'] = list(model.keys())
    except Exception as e:
        checks['model_loadable'] = False
        checks['error'] = str(e)

    return checks
```

### Performance Monitoring Commands

```bash
# System monitoring
htop
nvidia-smi
iostat -x 1

# Container monitoring
docker stats
kubectl top pods

# Application monitoring
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Log monitoring
tail -f logs/app.log | jq .
journalctl -u llm-safety -f
```

### Recovery Procedures

#### Model Rollback

```bash
#!/bin/bash
# rollback_model.sh

BACKUP_DIR="/app/models/backup"
CURRENT_DIR="/app/models"

# Find latest backup
LATEST_BACKUP=$(ls -t $BACKUP_DIR | head -n1)

if [ -n "$LATEST_BACKUP" ]; then
    echo "Rolling back to $LATEST_BACKUP"

    # Backup current
    mv $CURRENT_DIR $CURRENT_DIR.failed

    # Restore backup
    cp -r $BACKUP_DIR/$LATEST_BACKUP $CURRENT_DIR

    # Restart service
    systemctl restart llm-safety

    echo "Rollback completed"
else
    echo "No backup found"
    exit 1
fi
```

#### Service Recovery

```bash
#!/bin/bash
# recover_service.sh

echo "Starting service recovery..."

# Stop service
kubectl scale deployment llm-safety-deployment --replicas=0

# Wait for pods to terminate
kubectl wait --for=delete pod -l app=llm-safety --timeout=60s

# Start service
kubectl scale deployment llm-safety-deployment --replicas=3

# Wait for readiness
kubectl wait --for=condition=ready pod -l app=llm-safety --timeout=300s

echo "Service recovery completed"
```

## Conclusion

This deployment guide provides comprehensive instructions for deploying the LLM Safety system in production. Key points to remember:

1. **Security First**: Always implement proper authentication, validation, and encryption
2. **Monitor Everything**: Set up comprehensive monitoring and alerting
3. **Plan for Scale**: Design with horizontal scaling in mind
4. **Test Thoroughly**: Implement proper testing and validation pipelines
5. **Plan Recovery**: Have rollback and recovery procedures ready

For additional support or questions, refer to the project documentation or create an issue in the repository.
