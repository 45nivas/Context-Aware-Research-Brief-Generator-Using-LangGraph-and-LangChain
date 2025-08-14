# API Reference

Complete API reference for the Research Brief Generator.

## Base URL

```
https://your-deployment-url.com
```

For local development:
```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, consider implementing:
- API key authentication
- OAuth 2.0
- JWT tokens

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Brief Generation**: 10 requests per minute per IP
- **Status Checks**: 60 requests per minute per IP

## Content Types

All requests and responses use `application/json` content type.

## Error Handling

All error responses follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {},
    "timestamp": "2024-12-01T12:00:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `LLM_ERROR` | 503 | LLM service unavailable |
| `SEARCH_ERROR` | 503 | Search service unavailable |

## Endpoints

### Health Check

Check API health and status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-01T12:00:00Z",
  "database": "connected",
  "llm_models": "available",
  "search_services": "operational"
}
```

### Generate Research Brief

Generate a comprehensive research brief on a given topic.

**Endpoint:** `POST /brief`

**Request Body:**
```json
{
  "query": "Impact of artificial intelligence on software development",
  "depth": "standard",
  "user_id": "user123",
  "context": {
    "previous_queries": ["AI in healthcare", "Machine learning basics"],
    "preferences": {
      "focus_areas": ["technical", "business"],
      "excluded_sources": ["social_media"]
    }
  },
  "search_filters": {
    "date_range": "1y",
    "source_types": ["academic", "news", "reports"],
    "language": "en"
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Research topic or question |
| `depth` | enum | Yes | One of: `quick`, `standard`, `comprehensive` |
| `user_id` | string | No | User identifier for context tracking |
| `context` | object | No | User context and preferences |
| `search_filters` | object | No | Search customization options |

**Depth Levels:**

- **Quick** (5-10 minutes): Basic overview with 3-5 sources
- **Standard** (15-25 minutes): Comprehensive analysis with 8-12 sources
- **Comprehensive** (30-45 minutes): In-depth research with 15+ sources

**Response:**
```json
{
  "id": "brief_12345",
  "title": "Impact of Artificial Intelligence on Software Development",
  "query": "Impact of artificial intelligence on software development",
  "summary": "Comprehensive analysis of AI's transformative impact...",
  "key_findings": [
    "AI tools increase developer productivity by 25-40%",
    "Code generation and review are primary AI applications",
    "Concerns about job displacement are largely unfounded"
  ],
  "sources": [
    {
      "title": "AI in Software Development: A 2024 Survey",
      "url": "https://example.com/survey",
      "type": "report",
      "relevance_score": 0.95,
      "summary": "Large-scale survey of 10,000 developers...",
      "key_quotes": ["Quote 1", "Quote 2"],
      "publication_date": "2024-03-15"
    }
  ],
  "tags": ["artificial-intelligence", "software-development", "productivity"],
  "created_at": "2024-12-01T12:00:00Z",
  "updated_at": "2024-12-01T12:25:00Z",
  "user_id": "user123",
  "depth": "standard",
  "status": "completed",
  "processing_time": 1247.5,
  "metadata": {
    "sources_found": 12,
    "sources_processed": 10,
    "llm_calls": 15,
    "search_queries": 5
  }
}
```

### Get Brief Status

Check the status of a brief generation request.

**Endpoint:** `GET /brief/{brief_id}/status`

**Response:**
```json
{
  "id": "brief_12345",
  "status": "processing",
  "progress": 65,
  "current_step": "source_summarization",
  "estimated_remaining": 384,
  "created_at": "2024-12-01T12:00:00Z",
  "steps": [
    {
      "name": "context_summarization",
      "status": "completed",
      "duration": 45.2
    },
    {
      "name": "planning",
      "status": "completed", 
      "duration": 78.3
    },
    {
      "name": "search",
      "status": "completed",
      "duration": 234.1
    },
    {
      "name": "content_fetching",
      "status": "completed",
      "duration": 456.7
    },
    {
      "name": "source_summarization",
      "status": "processing",
      "duration": 289.4
    },
    {
      "name": "synthesis",
      "status": "pending"
    },
    {
      "name": "post_processing",
      "status": "pending"
    }
  ]
}
```

**Status Values:**
- `pending`: Request queued
- `processing`: Generation in progress
- `completed`: Successfully completed
- `failed`: Generation failed
- `cancelled`: Request cancelled

### Get Brief Details

Retrieve a previously generated brief.

**Endpoint:** `GET /brief/{brief_id}`

**Response:** Same as brief generation response.

### Update Brief

Update or regenerate specific sections of a brief.

**Endpoint:** `PATCH /brief/{brief_id}`

**Request Body:**
```json
{
  "sections_to_update": ["summary", "key_findings"],
  "additional_context": "Focus more on enterprise applications",
  "new_sources": ["https://example.com/new-research"]
}
```

### Delete Brief

Delete a brief from the system.

**Endpoint:** `DELETE /brief/{brief_id}`

**Response:**
```json
{
  "message": "Brief deleted successfully",
  "deleted_at": "2024-12-01T12:30:00Z"
}
```

### User History

Get all briefs for a specific user.

**Endpoint:** `GET /user/{user_id}/history`

**Query Parameters:**
- `limit`: Number of results (default: 20, max: 100)
- `offset`: Pagination offset (default: 0)
- `order_by`: Sort field (default: "created_at")
- `order`: Sort direction ("asc" or "desc", default: "desc")

**Response:**
```json
{
  "briefs": [
    {
      "id": "brief_12345",
      "title": "Impact of AI on Software Development",
      "query": "Impact of artificial intelligence on software development",
      "status": "completed",
      "created_at": "2024-12-01T12:00:00Z",
      "depth": "standard",
      "tags": ["ai", "software-development"]
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

### User Context

Get or update user context and preferences.

**Get Context:**
**Endpoint:** `GET /user/{user_id}/context`

**Update Context:**
**Endpoint:** `PUT /user/{user_id}/context`

**Request Body:**
```json
{
  "preferences": {
    "focus_areas": ["technical", "business"],
    "excluded_sources": ["social_media"],
    "preferred_depth": "standard",
    "output_format": "structured"
  },
  "domain_expertise": ["software_engineering", "ai_ml"],
  "research_goals": ["staying_current", "competitive_analysis"]
}
```

### Search Suggestions

Get search suggestions based on query.

**Endpoint:** `GET /suggest`

**Query Parameters:**
- `q`: Partial query string
- `limit`: Number of suggestions (default: 5, max: 10)

**Response:**
```json
{
  "suggestions": [
    "artificial intelligence in healthcare",
    "artificial intelligence ethics",
    "artificial intelligence market trends"
  ]
}
```

### Metrics and Analytics

Get system metrics and usage analytics.

**Endpoint:** `GET /metrics`

**Response:**
```json
{
  "requests_total": 15420,
  "requests_today": 342,
  "avg_processing_time": 1245.6,
  "success_rate": 0.987,
  "popular_topics": [
    {"topic": "artificial intelligence", "count": 1250},
    {"topic": "climate change", "count": 890}
  ],
  "system_health": {
    "database": "healthy",
    "llm_services": "healthy",
    "search_services": "healthy"
  }
}
```

## Webhooks

Register webhooks to receive notifications about brief completion.

**Register Webhook:**
**Endpoint:** `POST /webhooks`

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["brief.completed", "brief.failed"],
  "secret": "webhook_secret_for_verification"
}
```

**Webhook Payload:**
```json
{
  "event": "brief.completed",
  "brief_id": "brief_12345",
  "user_id": "user123",
  "timestamp": "2024-12-01T12:25:00Z",
  "data": {
    "title": "Brief Title",
    "status": "completed",
    "processing_time": 1247.5
  }
}
```

## SDKs and Libraries

### Python SDK

```bash
pip install research-brief-sdk
```

```python
from research_brief import ResearchBriefClient

client = ResearchBriefClient(api_url="https://api.researchbrief.ai")

# Generate brief
brief = await client.generate_brief(
    query="AI in healthcare",
    depth="standard",
    user_id="user123"
)

# Check status
status = await client.get_status(brief.id)

# Get user history
history = await client.get_user_history("user123")
```

### JavaScript SDK

```bash
npm install research-brief-js
```

```javascript
import { ResearchBriefClient } from 'research-brief-js';

const client = new ResearchBriefClient({
  apiUrl: 'https://api.researchbrief.ai'
});

// Generate brief
const brief = await client.generateBrief({
  query: 'AI in healthcare',
  depth: 'standard',
  userId: 'user123'
});

// Check status
const status = await client.getStatus(brief.id);
```

## OpenAPI Specification

The complete OpenAPI/Swagger specification is available at:
- JSON: `GET /openapi.json`
- Interactive docs: `GET /docs`
- ReDoc: `GET /redoc`

## Examples

### Curl Examples

**Generate a quick brief:**
```bash
curl -X POST "https://api.researchbrief.ai/brief" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Latest trends in renewable energy",
    "depth": "quick",
    "user_id": "demo_user"
  }'
```

**Check brief status:**
```bash
curl "https://api.researchbrief.ai/brief/brief_12345/status"
```

**Get user history:**
```bash
curl "https://api.researchbrief.ai/user/demo_user/history?limit=10"
```

### Response Times

Expected response times by depth level:

| Depth | Avg Time | Sources | LLM Calls |
|-------|----------|---------|-----------|
| Quick | 5-10 min | 3-5 | 5-8 |
| Standard | 15-25 min | 8-12 | 12-18 |
| Comprehensive | 30-45 min | 15+ | 20+ |

## Best Practices

1. **Use appropriate depth levels** for your use case
2. **Provide user context** for better personalization
3. **Implement proper error handling** for all requests
4. **Cache results** when appropriate
5. **Monitor rate limits** to avoid throttling
6. **Use webhooks** for long-running requests
7. **Include source filters** to improve relevance

## Support

- **Documentation**: [docs.researchbrief.ai](https://docs.researchbrief.ai)
- **API Status**: [status.researchbrief.ai](https://status.researchbrief.ai)
- **Support Email**: api-support@researchbrief.ai
- **GitHub Issues**: [github.com/username/research-brief-generator/issues](https://github.com/username/research-brief-generator/issues)
