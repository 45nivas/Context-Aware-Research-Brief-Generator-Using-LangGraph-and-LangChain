# Deployment Guide

This guide provides step-by-step instructions for deploying the Research Brief Generator to various platforms.

## 🚀 Quick Deploy Options

### 1. Railway (Recommended)

Railway provides the easiest deployment with automatic HTTPS and domain management.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy from GitHub (recommended)
railway up

# Or deploy from local directory
cd research-brief-generator
railway up
```

**Environment Variables to Set in Railway:**
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
DATABASE_URL=sqlite+aiosqlite:///./data/research_briefs.db
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=research-brief-generator
API_HOST=0.0.0.0
API_PORT=$PORT
```

### 2. Render

Render offers free tier with automatic deployments from GitHub.

1. **Connect Repository**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"

2. **Configure Service**
   ```yaml
   Name: research-brief-generator
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python -m app.api
   ```

3. **Set Environment Variables**
   Add all required API keys in the Render dashboard.

### 3. AWS ECS with Fargate

For production deployments with full control.

```bash
# Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

docker build -t research-brief-generator .
docker tag research-brief-generator:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/research-brief-generator:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/research-brief-generator:latest

# Deploy with Terraform (see terraform/ directory)
cd terraform
terraform init
terraform plan
terraform apply
```

### 4. Google Cloud Run

Serverless deployment with automatic scaling.

```bash
# Enable required services
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Deploy with Cloud Build
gcloud run deploy research-brief-generator \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="OPENAI_API_KEY=your_key,ANTHROPIC_API_KEY=your_key"
```

### 5. Azure Container Instances

Quick deployment to Azure.

```bash
# Create resource group
az group create --name research-brief-rg --location eastus

# Deploy container
az container create \
  --resource-group research-brief-rg \
  --name research-brief-generator \
  --image your-registry/research-brief-generator:latest \
  --environment-variables OPENAI_API_KEY=your_key ANTHROPIC_API_KEY=your_key \
  --ports 8000 \
  --dns-name-label research-brief-unique
```

## 🐳 Docker Deployment

### Local Docker

```bash
# Build image
docker build -t research-brief-generator .

# Run container
docker run -d \
  --name research-brief \
  -p 8000:8000 \
  --env-file .env \
  research-brief-generator
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔧 Configuration Management

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key | - |
| `TAVILY_API_KEY` | Yes | Tavily search API key | - |
| `LANGSMITH_API_KEY` | No | LangSmith tracing key | - |
| `DATABASE_URL` | No | Database connection string | `sqlite+aiosqlite:///./research_briefs.db` |
| `API_HOST` | No | API server host | `0.0.0.0` |
| `API_PORT` | No | API server port | `8000` |
| `DEBUG` | No | Enable debug mode | `false` |

### Health Checks

All deployments should include health checks:

```bash
# Health check endpoint
curl https://your-domain.com/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-12-01T12:00:00Z",
  "database": "connected",
  "llm_models": "available"
}
```

## 🔒 Security Considerations

### API Keys Management

1. **Never commit API keys to version control**
2. **Use environment variables or secret managers**
3. **Rotate keys regularly**
4. **Monitor API usage and costs**

### Network Security

```bash
# Enable HTTPS (Railway/Render do this automatically)
# For custom deployments, use Let's Encrypt:

certbot --nginx -d your-domain.com
```

### Rate Limiting

Configure rate limiting in production:

```python
# Add to FastAPI app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/brief")
@limiter.limit("10/minute")
async def generate_brief(request: Request, ...):
    ...
```

## 📊 Monitoring & Observability

### LangSmith Tracing

Enable comprehensive tracing in production:

```bash
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=research-brief-generator-prod
```

### Application Monitoring

#### Using Prometheus + Grafana

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'research-brief-api'
    static_configs:
      - targets: ['api:8000']
```

#### Custom Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

brief_requests = Counter('brief_requests_total', 'Total brief requests')
brief_duration = Histogram('brief_duration_seconds', 'Brief generation duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 🔧 Database Management

### SQLite (Default)

For small to medium deployments:

```bash
# Database will be created automatically
# Backup database
cp research_briefs.db research_briefs_backup_$(date +%Y%m%d).db
```

### PostgreSQL (Production)

For high-scale deployments:

```bash
# Set database URL
DATABASE_URL=postgresql+asyncpg://user:password@host:port/database

# Run migrations (if using Alembic)
alembic upgrade head
```

## 🚨 Troubleshooting

### Common Deployment Issues

1. **API Keys Not Working**
   ```bash
   # Check environment variables
   curl https://your-domain.com/health
   ```

2. **Database Connection Issues**
   ```bash
   # Check database permissions and path
   ls -la research_briefs.db
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   # Increase container memory if needed
   ```

4. **Timeout Issues**
   ```bash
   # Check LLM API response times
   # Increase timeout values in config
   ```

### Performance Optimization

1. **Enable HTTP/2 and compression**
2. **Use CDN for static assets**
3. **Implement Redis caching for frequent requests**
4. **Scale horizontally with load balancer**

## 📈 Scaling Considerations

### Horizontal Scaling

```bash
# Railway
railway scale --replicas 3

# AWS ECS
aws ecs update-service --cluster default --service research-brief-service --desired-count 3

# Kubernetes
kubectl scale deployment research-brief-generator --replicas=3
```

### Database Scaling

1. **Read Replicas**: For read-heavy workloads
2. **Sharding**: For write-heavy workloads
3. **Connection Pooling**: Use pgbouncer or similar

### Cost Optimization

1. **Monitor LLM token usage**
2. **Implement result caching**
3. **Use spot instances where possible**
4. **Set up billing alerts**

## 🎯 Production Checklist

- [ ] API keys configured securely
- [ ] Health checks enabled
- [ ] HTTPS configured
- [ ] Rate limiting implemented
- [ ] Monitoring and alerting set up
- [ ] Database backups configured
- [ ] Error logging enabled
- [ ] Performance testing completed
- [ ] Security scanning performed
- [ ] Documentation updated

## 📞 Support

For deployment issues:
- Check the [troubleshooting guide](README.md#troubleshooting)
- Open an issue on [GitHub](https://github.com/username/research-brief-generator/issues)
- Contact support: deploy@researchbrief.ai
