# Outfit Recommender API

A Style & Occasion-Based Outfit Recommendation System using CLIP embeddings and machine learning classification for personalized fashion suggestions.

### Frontend Application
The frontend for this API can be found at [outfit-reco](https://github.com/kapadilla/outfit-reco).

## Project Overview

This API provides intelligent fashion recommendations based on natural language queries. It combines:

- **CLIP (ViT-B/32)** for visual and text embedding
- **Logistic Regression** with class balancing for usage category classification
- **SMOTE + Class Weights** for handling imbalanced data
- **Cosine Similarity** for semantic matching and ranking

Based on the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle.

## Features

- 🔍 Natural language search for outfit recommendations
- 🎯 Automatic usage category prediction (Casual, Formal, Sports, Ethnic, etc.)
- ⚖️ Balanced predictions across all categories (not biased toward Casual)
- 📊 Ranked results by visual similarity
- 🖼️ Product metadata and image serving
- 📚 Interactive API documentation

## Project Structure

```
outfit-reco_api/
├── backend/
│   ├── config/           # Configuration and settings
│   ├── models/           # Pydantic schemas for validation
│   ├── routes/           # API endpoint handlers
│   ├── services/         # Business logic (ML & recommendations)
│   ├── api.py           # Main FastAPI application
│   └── requirements.txt
├── data/
│   ├── images/          # Product images
│   └── styles.csv       # Product metadata
├── model/               # Pre-trained ML models
│   ├── usage_classifier.joblib
│   ├── label_encoder.joblib
│   ├── image_embeddings.npy
│   └── image_ids.csv
├── scripts/
│   ├── download_dataset.py  # Kaggle dataset downloader
│   └── train_models.py      # Model training with class balancing
├── .env                 # Environment variables (not in git)
└── .env.example        # Environment template
```


## Setup Instructions

### 1. Clone the Repository

```bash
cd outfit-reco_api
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env` and add your Kaggle API token:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_kaggle_token_here` with your actual token.

### 5. Download Dataset

```bash
python scripts/download_dataset.py
```

This will:

- Download the Fashion Product Images Dataset from Kaggle (~15GB)
- Extract `styles.csv` to `data/`
- Copy product images to `data/images/`

### 6. Run the API

**Important**: You must run the server from the `backend/` directory.

```bash
cd backend
python -m uvicorn api:app --reload
```

Or start with explicit host/port:

```bash
cd backend
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The API will be available at:

- **API**: http://127.0.0.1:8000
- **Interactive Docs**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc


## API Endpoints

### Quick Reference

| Endpoint           | Method | Description                   |
| ------------------ | ------ | ----------------------------- |
| `/`                | GET    | Root endpoint with API info   |
| `/health`          | GET    | Health check and status       |
| `/recommend`       | GET    | Get outfit recommendations    |
| `/images/{id}.jpg` | GET    | Serve product images          |
| `/docs`            | GET    | Interactive API documentation |

### GET `/recommend`

Get personalized outfit recommendations based on natural language queries.

**Parameters:**

- `q` (required, string): Search query describing desired outfit
- `top_k` (optional, integer): Number of results (1-50, default: 12)

**Request:**

```bash
curl "http://localhost:8000/recommend?q=casual%20blue%20jeans&top_k=10"
```

**Response:**

```json
{
  "query": "casual blue jeans",
  "predicted_usage": "Casual",
  "results": [
    {
      "id": 15970,
      "product": "Roadster Men Blue Denim Jeans",
      "color": "Blue",
      "usage": "Casual",
      "score": 0.8523
    }
  ],
  "total_results": 10
}
```

### GET `/health`

Check API operational status, model availability, and data integrity.

**Request:**

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "data_available": true,
  "missing_files": null
}
```

### GET `/images/{image_id}`

Retrieve product images by ID.

**Request:**

```bash
curl http://localhost:8000/images/15970.jpg -o product.jpg
```

**HTML Usage:**

```html
<img src="http://localhost:8000/images/15970.jpg" alt="Product" />
```

### Complete Documentation

For detailed API documentation including all parameters, response schemas, error codes, and usage examples, see [API_DOCUMENTATION.md](refs/API_DOCUMENTATION.md).

## Development

### Project Architecture

The API follows a modular architecture with clear separation of concerns:

1. **Config Layer** (`backend/config/`)

   - Centralized settings management
   - Environment variable loading
   - Path validation

2. **Models Layer** (`backend/models/`)

   - Pydantic schemas for request/response validation
   - Type safety and automatic documentation

3. **Services Layer** (`backend/services/`)

   - `MLService`: Model loading and inference
   - `RecommendationService`: Recommendation logic and ranking

4. **Routes Layer** (`backend/routes/`)
   - HTTP endpoint handlers
   - Request validation and error handling

### Adding New Features

To add a new endpoint:

1. Define schemas in `backend/models/schemas.py`
2. Implement logic in appropriate service
3. Create route handler in `backend/routes/`
4. Register router in `backend/api.py`

## Dataset Information

The Fashion Product Images Dataset contains:

- **44,424 products** with images and metadata
- **10 metadata columns**: id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName
- **Usage categories**: Casual, Formal, Sports, Ethnic, Party, Smart Casual, Travel, Home

## Technologies Used

- **FastAPI**: Modern web framework for building APIs
- **CLIP**: OpenAI's vision-language model
- **PyTorch**: Deep learning framework
- **Scikit-learn**: Machine learning library
- **imbalanced-learn**: SMOTE oversampling for class balancing
- **Pandas**: Data manipulation
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## Retraining the Model

To retrain the model with class balancing:

```bash
# Full dataset (takes ~3 hours on CPU)
python scripts/train_models.py

# Quick test with 1000 samples
python scripts/train_models.py --sample-size 1000

# Without SMOTE (class weights only)
python scripts/train_models.py --no-smote
```

After retraining, restart the API server to load the new model.


## Troubleshooting

### "Missing required files" error

Run the dataset download script:

```bash
python scripts/download_dataset.py
```

### "Models not loaded" error

Check that all model files exist in the `model/` directory:

- `usage_classifier.joblib`
- `label_encoder.joblib`
- `image_embeddings.npy`
- `image_ids.csv`

### CUDA/GPU issues

If you don't have a GPU, ensure `DEVICE=cpu` in your `.env` file.

## License

This project uses the Fashion Product Images Dataset from Kaggle, subject to their terms and conditions.

## Contributors

- Bagazin, Christine Mae
- Lapaz, Jermel
- Padilla, Kimberly

University of the Philippines Cebu - Machine Learning Final Project (December 2025)
