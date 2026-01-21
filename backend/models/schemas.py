"""
Pydantic models for request/response validation and data schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ProductItem(BaseModel):
    """Schema for a single product recommendation."""

    id: int = Field(..., description="Product ID")
    product: str = Field(default="", description="Product display name")
    color: str = Field(default="", description="Base color of the product")
    usage: str = Field(default="", description="Usage category (e.g., Casual, Formal)")
    score: float = Field(..., description="Similarity score (0-1)", ge=0, le=1)
    image_url: str = Field(default="", description="URL path to the product image")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 12345,
                "product": "Blue Denim Jeans",
                "color": "Blue",
                "usage": "Casual",
                "score": 0.85,
                "image_url": "/images/12345.jpg",
            }
        }


class RecommendationRequest(BaseModel):
    """Schema for recommendation request."""

    q: str = Field(
        ..., description="Search query describing desired outfit", min_length=1
    )
    top_k: Optional[int] = Field(
        default=12, description="Number of recommendations to return", ge=1, le=50
    )

    class Config:
        json_schema_extra = {
            "example": {"q": "casual blue jeans for summer", "top_k": 12}
        }


class RecommendationResponse(BaseModel):
    """Schema for recommendation response with pagination support."""

    query: str = Field(..., description="Original search query")
    predicted_usage: Optional[str] = Field(None, description="Predicted usage category")
    results: List[ProductItem] = Field(..., description="List of recommended products")
    total_results: int = Field(..., description="Number of results in current page")
    
    # Pagination fields
    page: int = Field(default=1, description="Current page number")
    limit: int = Field(default=12, description="Items per page")
    total_matching: int = Field(default=0, description="Total matching products before pagination")
    total_pages: int = Field(default=1, description="Total number of pages")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "casual blue jeans for summer",
                "predicted_usage": "Casual",
                "results": [
                    {
                        "id": 12345,
                        "product": "Blue Denim Jeans",
                        "color": "Blue",
                        "usage": "Casual",
                        "score": 0.85,
                        "image_url": "/images/12345.jpg",
                    }
                ],
                "total_results": 12,
                "page": 1,
                "limit": 12,
                "total_matching": 150,
                "total_pages": 13,
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether ML models are loaded")
    data_available: bool = Field(..., description="Whether dataset is available")
    missing_files: Optional[List[str]] = Field(
        None, description="List of missing required files"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True,
                "data_available": True,
                "missing_files": [],
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "An error occurred while processing your request",
                "error_type": "InternalServerError",
            }
        }
