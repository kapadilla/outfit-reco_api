# API Endpoint Documentation

Complete reference guide for the Outfit Recommender API endpoints.

---

## Table of Contents

1. [Base URL](#base-url)
2. [Authentication](#authentication)
3. [Response Format](#response-format)
4. [Endpoints](#endpoints)
   - [GET /](#get-)
   - [GET /health](#get-health)
   - [GET /recommend](#get-recommend)
   - [GET /images/{image_id}](#get-imagesimage_id)
5. [Error Handling](#error-handling)
6. [Usage Examples](#usage-examples)

---

## Base URL

```
http://localhost:8000
```

For production deployment, replace with your actual domain.

---

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## Response Format

All API responses return JSON data (except for image endpoints which return binary data).

**Success Response Structure:**

```json
{
  "field1": "value1",
  "field2": "value2"
}
```

**Error Response Structure:**

```json
{
  "detail": "Error message describing what went wrong",
  "error_type": "ErrorType"
}
```

---

## Endpoints

### GET `/`

**Description:**  
Root endpoint that provides basic API information and quick navigation links.

**Purpose:**

- Welcome page for the API
- Quick reference to available endpoints
- API version information

**Parameters:**  
None

**Request Headers:**  
None required

**Response:**

**Status Code:** `200 OK`

**Response Body:**

```json
{
  "message": "Welcome to the Outfit Recommender API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "recommend": "/recommend?q=your+query+here"
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `message` | string | Welcome message |
| `version` | string | Current API version |
| `docs` | string | Path to interactive documentation |
| `health` | string | Path to health check endpoint |
| `recommend` | string | Example path to recommendation endpoint |

**Example Request:**

```bash
curl http://localhost:8000/
```

**Example Response:**

```json
{
  "message": "Welcome to the Outfit Recommender API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "recommend": "/recommend?q=your+query+here"
}
```

**Use Cases:**

- Quick API status verification
- Getting started with API navigation
- Checking API version

---

### GET `/health`

**Description:**  
Health check endpoint that verifies the API's operational status, including model availability and data accessibility.

**Purpose:**

- Monitor API health status
- Verify ML models are loaded correctly
- Check dataset availability
- Identify missing required files

**Parameters:**  
None

**Request Headers:**  
None required

**Response:**

**Status Code:** `200 OK`

**Response Body:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "data_available": true,
  "missing_files": null
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall API status (`"ok"` or `"error"`) |
| `model_loaded` | boolean | Whether ML models are successfully loaded |
| `data_available` | boolean | Whether all required data files are present |
| `missing_files` | array\|null | List of missing required files (null if none missing) |

**Example Request:**

```bash
curl http://localhost:8000/health
```

**Example Response (Healthy):**

```json
{
  "status": "ok",
  "model_loaded": true,
  "data_available": true,
  "missing_files": null
}
```

**Example Response (Missing Data):**

```json
{
  "status": "ok",
  "model_loaded": false,
  "data_available": false,
  "missing_files": ["styles_csv", "image_embeddings"]
}
```

**Use Cases:**

- Health monitoring in production
- Pre-deployment verification
- Troubleshooting setup issues
- Automated health checks in CI/CD pipelines

---

### GET `/recommend`

**Description:**  
Main recommendation endpoint that returns personalized outfit suggestions based on a natural language query. Uses CLIP embeddings to understand the query, predicts usage category, and ranks products by visual similarity.

**Purpose:**

- Get outfit recommendations from text descriptions
- Find fashion items matching specific criteria
- Discover products by occasion, style, or attributes

**Parameters:**

| Parameter | Type    | Required | Default | Constraints    | Description                                        |
| --------- | ------- | -------- | ------- | -------------- | -------------------------------------------------- |
| `q`       | string  | Yes      | -       | min_length=1   | Search query describing desired outfit or occasion |
| `top_k`   | integer | No       | 12      | 1 ≤ value ≤ 50 | Number of recommendations to return                |

**Query Parameter Details:**

**`q` (Query String):**

- Natural language description of what you're looking for
- Can include: style, color, occasion, usage, season, etc.
- Examples: "casual blue jeans", "formal business attire", "summer party dress"

**`top_k` (Top K Results):**

- Controls how many recommendations are returned
- Higher values return more results
- Lower values for more focused recommendations
- Default: 12 items

**Request Headers:**  
None required

**Response:**

**Status Code:** `200 OK`

**Response Body:**

```json
{
  "query": "casual blue jeans for summer",
  "predicted_usage": "Casual",
  "results": [
    {
      "id": 15970,
      "product": "Roadster Men Blue Denim Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8523,
      "image_url": "/images/15970.jpg"
    },
    {
      "id": 39403,
      "product": "Wrangler Men Blue Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8401,
      "image_url": "/images/39403.jpg"
    }
  ],
  "total_results": 2
}
```

**Response Fields:**

| Field             | Type         | Description                                                  |
| ----------------- | ------------ | ------------------------------------------------------------ |
| `query`           | string       | Original search query from the request                       |
| `predicted_usage` | string\|null | Predicted usage category (e.g., Casual, Formal, Sports)      |
| `results`         | array        | List of recommended products (see Product Item schema below) |
| `total_results`   | integer      | Number of results returned                                   |

**Product Item Schema:**

| Field       | Type    | Description                                                        |
| ----------- | ------- | ------------------------------------------------------------------ |
| `id`        | integer | Unique product identifier                                          |
| `product`   | string  | Full product display name                                          |
| `color`     | string  | Base color of the product                                          |
| `usage`     | string  | Usage category (Casual, Formal, Sports, etc.)                      |
| `score`     | float   | Similarity score between 0 and 1 (higher = more relevant)          |
| `image_url` | string  | Relative URL path to the product image (e.g., `/images/12345.jpg`) |

**Example Requests:**

**Basic Query:**

```bash
curl "http://localhost:8000/recommend?q=casual%20blue%20jeans"
```

**Query with Custom Result Count:**

```bash
curl "http://localhost:8000/recommend?q=formal%20business%20shirt&top_k=20"
```

**Complex Query:**

```bash
curl "http://localhost:8000/recommend?q=ethnic%20dress%20for%20party%20red%20color"
```

**Example Response:**

```json
{
  "query": "casual blue jeans for summer",
  "predicted_usage": "Casual",
  "results": [
    {
      "id": 15970,
      "product": "Roadster Men Blue Denim Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8523456,
      "image_url": "/images/15970.jpg"
    },
    {
      "id": 39403,
      "product": "Wrangler Men Blue Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8401234,
      "image_url": "/images/39403.jpg"
    },
    {
      "id": 12347,
      "product": "United Colors of Benetton Men Blue Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8256789,
      "image_url": "/images/12347.jpg"
    },
    {
      "id": 28232,
      "product": "Lee Men Blue Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8134567,
      "image_url": "/images/28232.jpg"
    },
    {
      "id": 44551,
      "product": "Flying Machine Men Blue Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8023456,
      "image_url": "/images/44551.jpg"
    }
  ],
  "total_results": 5
}
```

**Error Responses:**

**Missing Query Parameter (400 Bad Request):**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["query", "q"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

**Invalid top_k Value (422 Unprocessable Entity):**

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["query", "top_k"],
      "msg": "Input should be greater than or equal to 1",
      "input": "0"
    }
  ]
}
```

**Service Error (500 Internal Server Error):**

```json
{
  "detail": "An error occurred while processing your request: [error details]",
  "error_type": "InternalServerError"
}
```

**Models Not Loaded (503 Service Unavailable):**

```json
{
  "detail": "Service unavailable: Models not loaded. Call load_models() first.",
  "error_type": "ServiceUnavailable"
}
```

**Query Examples and Expected Results:**

| Query                      | Predicted Usage | Expected Results                           |
| -------------------------- | --------------- | ------------------------------------------ |
| `"casual blue jeans"`      | Casual          | Blue denim jeans, casual pants             |
| `"formal business attire"` | Formal          | Business suits, formal shirts, dress pants |
| `"sports running shoes"`   | Sports          | Running shoes, athletic footwear           |
| `"ethnic dress for party"` | Party/Ethnic    | Traditional dresses, ethnic wear           |
| `"summer beach outfit"`    | Casual          | Light clothing, casual summer wear         |
| `"winter jacket warm"`     | Casual/Sports   | Winter jackets, warm outerwear             |

**Use Cases:**

- E-commerce product search
- Personal styling recommendations
- Outfit discovery by occasion
- Fashion catalog exploration
- Virtual wardrobe management

**Performance Notes:**

- First request may be slower due to model initialization
- Subsequent requests are faster (models remain loaded)
- Response time depends on `top_k` value (more results = slightly longer)
- Average response time: 200-500ms for top_k=12

---

### GET `/images/{image_id}`

**Description:**  
Serves product images as static files. Returns the actual image file for a given product ID.

**Purpose:**

- Display product images in frontend applications
- Preview recommended items
- Image caching and CDN integration

**Path Parameters:**

| Parameter  | Type           | Required | Description                            |
| ---------- | -------------- | -------- | -------------------------------------- |
| `image_id` | integer/string | Yes      | Product ID (must match image filename) |

**File Format:**  
Images are served with the following naming convention:

- Format: `{image_id}.jpg`
- Example: `15970.jpg`, `39403.jpg`

**Request Headers:**  
None required (standard HTTP GET)

**Response:**

**Status Code:** `200 OK` (if image exists)

**Response Headers:**

```
Content-Type: image/jpeg
Content-Length: [file size in bytes]
```

**Response Body:**  
Binary image data (JPEG format)

**Error Response:**

**Status Code:** `404 Not Found` (if image doesn't exist)

**Example Requests:**

**Direct Browser Access:**

```
http://localhost:8000/images/15970.jpg
```

**cURL:**

```bash
curl http://localhost:8000/images/15970.jpg --output product.jpg
```

**HTML Image Tag:**

```html
<img src="http://localhost:8000/images/15970.jpg" alt="Product 15970" />
```

**JavaScript Fetch:**

```javascript
fetch("http://localhost:8000/images/15970.jpg")
  .then((response) => response.blob())
  .then((blob) => {
    const imageUrl = URL.createObjectURL(blob);
    document.getElementById("product-image").src = imageUrl;
  });
```

**React Example (Using image_url from response):**

```jsx
const API_BASE = "http://localhost:8000";

function ProductCard({ item }) {
  return (
    <div className="product-card">
      <img
        src={`${API_BASE}${item.image_url}`}
        alt={item.product}
        onError={(e) => (e.target.src = "/placeholder.jpg")}
      />
      <h3>{item.product}</h3>
      <p>Color: {item.color}</p>
      <p>Usage: {item.usage}</p>
      <p>Score: {(item.score * 100).toFixed(1)}%</p>
    </div>
  );
}

function RecommendationResults({ results }) {
  return (
    <div className="results-grid">
      {results.map((item) => (
        <ProductCard key={item.id} item={item} />
      ))}
    </div>
  );
}
```

**Use Cases:**

- Displaying product images in recommendation results
- Building product galleries
- Image caching for offline access
- Integration with frontend frameworks
- Social media sharing

**Notes:**

- Images are served from the `data/images/` directory
- Only `.jpg` format is supported
- Images are not resized or optimized by the API (serve original)
- For production, consider using a CDN for better performance
- Missing images return 404 (handle gracefully in frontend)

---

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages.

### HTTP Status Codes

| Code | Meaning               | When It Occurs                                  |
| ---- | --------------------- | ----------------------------------------------- |
| 200  | OK                    | Request succeeded                               |
| 400  | Bad Request           | Invalid request parameters                      |
| 404  | Not Found             | Resource not found (e.g., image doesn't exist)  |
| 422  | Unprocessable Entity  | Validation error (e.g., invalid parameter type) |
| 500  | Internal Server Error | Server-side error during processing             |
| 503  | Service Unavailable   | ML models not loaded or service down            |

### Error Response Format

```json
{
  "detail": "Human-readable error message",
  "error_type": "ErrorType"
}
```

### Common Errors

**Validation Error (422):**

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["query", "q"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

**Service Unavailable (503):**

```json
{
  "detail": "Service unavailable: Models not loaded",
  "error_type": "ServiceUnavailable"
}
```

---

## Usage Examples

### Python with Requests

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get recommendations
params = {
    "q": "casual blue jeans for summer",
    "top_k": 10
}
response = requests.get(f"{BASE_URL}/recommend", params=params)
recommendations = response.json()

# Display results
print(f"Query: {recommendations['query']}")
print(f"Predicted Usage: {recommendations['predicted_usage']}")
print(f"\nTop {recommendations['total_results']} Recommendations:")

for item in recommendations['results']:
    print(f"  - {item['product']} (Score: {item['score']:.4f})")
    print(f"    ID: {item['id']}, Color: {item['color']}, Usage: {item['usage']}")
    print(f"    Image: {BASE_URL}{item['image_url']}")
```

### JavaScript with Fetch

```javascript
const BASE_URL = "http://localhost:8000";

// Get recommendations
async function getRecommendations(query, topK = 12) {
  const url = `${BASE_URL}/recommend?q=${encodeURIComponent(
    query
  )}&top_k=${topK}`;

  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    throw error;
  }
}

// Usage
getRecommendations("casual blue jeans", 10).then((data) => {
  console.log("Query:", data.query);
  console.log("Predicted Usage:", data.predicted_usage);
  console.log("Results:", data.results);
});
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Basic recommendation
curl "http://localhost:8000/recommend?q=casual%20blue%20jeans"

# With custom result count
curl "http://localhost:8000/recommend?q=formal%20shirt&top_k=20"

# Download product image
curl http://localhost:8000/images/15970.jpg --output product_15970.jpg

# Pretty print JSON response
curl "http://localhost:8000/recommend?q=sports%20shoes" | python -m json.tool
```

### Testing with HTTPie

```bash
# Install HTTPie: pip install httpie

# Health check
http GET localhost:8000/health

# Get recommendations
http GET localhost:8000/recommend q=="casual blue jeans" top_k==10

# Download image
http --download localhost:8000/images/15970.jpg
```

---

## Rate Limiting

Currently, the API does not implement rate limiting. For production deployment, consider adding rate limiting middleware.

---

## CORS Configuration

The API allows all origins (`*`) by default. For production, update `backend/config/settings.py`:

```python
CORS_ORIGINS = ["https://yourdomain.com"]  # Restrict to specific origins
```

---

## Interactive Documentation

Visit http://localhost:8000/docs for interactive API documentation where you can:

- View all endpoints
- Try out requests directly in the browser
- See request/response schemas
- Download OpenAPI specification

Alternative documentation: http://localhost:8000/redoc

---

## Support

For issues or questions:

- Check the `/health` endpoint for system status
- Review error messages in responses
- Ensure dataset is properly downloaded
- Verify all model files are present

---

**Last Updated:** December 18, 2025  
**API Version:** 1.0.0
