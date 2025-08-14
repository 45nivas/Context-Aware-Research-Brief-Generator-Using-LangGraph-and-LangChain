# 🚀 Deployment Guide

This guide covers deploying the Research Brief Generator to various cloud platforms.

## Quick Deploy Options

### Option 1: Render (Recommended - Free Tier)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yourusername/research-brief-generator)

1. **Fork this repository**
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Select this repository

3. **Configure Environment Variables:**
   ```bash
   GOOGLE_API_KEY=your_google_api_key
   SERPAPI_API_KEY=your_serpapi_key  
   LANGSMITH_API_KEY=your_langsmith_key
   ```

4. **Deploy:**
   - Render will automatically use `render.yaml`
   - Build takes ~5-10 minutes
   - Free tier includes 750 hours/month

### Option 2: Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/your-template)

1. **One-click deploy** with Railway template
2. **Set environment variables** in Railway dashboard
3. **Custom domain** available on Pro plan

### Option 3: Heroku

```bash
# Install Heroku CLI
npm install -g heroku

# Login and create app
heroku login
heroku create your-app-name

# Set environment variables
heroku config:set GOOGLE_API_KEY=your_key
heroku config:set SERPAPI_API_KEY=your_key
heroku config:set LANGSMITH_API_KEY=your_key

# Deploy
git push heroku main
```

### Option 4: Docker + Any Cloud

```bash
# Build Docker image
docker build -t research-brief-generator .

# Run locally
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key \
  -e SERPAPI_API_KEY=your_key \
  research-brief-generator

# Deploy to AWS/GCP/Azure using container services
```

## Environment Variables

### Required
- `GOOGLE_API_KEY`: Google Gemini API key
- `LANGSMITH_API_KEY`: For tracing and monitoring

### Optional (Free Tier)
- `SERPAPI_API_KEY`: Free tier: 100 searches/month
- `OLLAMA_BASE_URL`: For local Ollama models

### Production Settings
- `DATABASE_URL`: PostgreSQL for production
- `LANGCHAIN_TRACING_V2=true`: Enable tracing
- `DEBUG=false`: Disable debug mode

## Post-Deployment

### 1. Verify Deployment
```bash
curl https://your-app.onrender.com/health
```

### 2. Test API
```bash
curl -X POST https://your-app.onrender.com/brief \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "AI trends in 2025",
    "depth": 2,
    "follow_up": false,
    "user_id": "test_user"
  }'
```

### 3. Monitor with LangSmith
- Visit [LangSmith dashboard](https://smith.langchain.com)
- View traces and performance metrics
- Monitor token usage and costs

## Cost Optimization

### Free Tier Usage
- **Render**: 750 hours/month free
- **Google Gemini**: 15 requests/minute, 1M tokens/month
- **SerpAPI**: 100 searches/month
- **LangSmith**: 5K traces/month

### Scaling Strategy
1. **Start with free tiers**
2. **Monitor usage** with LangSmith
3. **Upgrade selectively** based on bottlenecks
4. **Use Ollama** for unlimited local processing

## Troubleshooting

### Common Issues

**Build Failures:**
```bash
# Check requirements.txt compatibility
pip install -r requirements.txt --dry-run
```

**API Timeout:**
```bash
# Increase timeout in production
export API_TIMEOUT=300
```

**Database Issues:**
```bash
# Use PostgreSQL for production
export DATABASE_URL=postgresql://user:pass@host:port/db
```

### Health Checks
```bash
# Basic health
curl https://your-app.onrender.com/health

# Detailed status
curl https://your-app.onrender.com/status

# API documentation
open https://your-app.onrender.com/docs
```

## Security

### API Keys
- Never commit API keys to Git
- Use environment variables
- Rotate keys regularly

### Database
- Use PostgreSQL in production
- Enable SSL connections
- Regular backups

### HTTPS
- All deployments use HTTPS by default
- Custom domains supported
- SSL certificates auto-renewed

## Monitoring

### LangSmith Tracing
- Real-time execution traces
- Token usage analytics
- Performance bottlenecks
- Error tracking

### Application Metrics
- Response times
- Success rates
- Resource usage
- Cost tracking

## Support

- 📧 **Issues**: [GitHub Issues](https://github.com/yourusername/research-brief-generator/issues)
- 📖 **Documentation**: [Full Docs](./README.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/research-brief-generator/discussions)
