"""
Outfit Recommender API - Main Application Entry Point

A Style & Occasion-Based Outfit Recommendation System using CLIP embeddings
and machine learning classification for personalized fashion suggestions.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from services import ml_service
from routes import health_router, recommendations_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Loads ML models on startup and performs cleanup on shutdown.
    """
    # Startup: Load ML models
    logger.info("Starting Outfit Recommender API...")
    try:
        ml_service.load_models()
        logger.info("✓ API startup complete")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        logger.warning(
            "API will start but recommendations will not work until models are loaded"
        )

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down API...")


# Initialize FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Mount static files for product images
if settings.IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(settings.IMAGES_DIR)), name="images")
    logger.info(f"✓ Mounted static images directory: {settings.IMAGES_DIR}")
else:
    logger.warning(f"Images directory not found: {settings.IMAGES_DIR}")

# Include API routes
app.include_router(health_router)
app.include_router(recommendations_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Outfit Recommender API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "recommend": "/recommend?q=your+query+here",
    }
