"""
Recommendation routes for outfit suggestions.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from models.schemas import RecommendationResponse
from services import recommendation_service
from config import settings

router = APIRouter(tags=["Recommendations"])


@router.get(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get Outfit Recommendations",
    description="""
    Get personalized outfit recommendations based on a text query with pagination support.
    
    The system uses CLIP embeddings to understand your query and matches it with 
    fashion products from our catalog. It predicts the usage category (e.g., Casual, 
    Formal) and returns the most similar items ranked by relevance.
    
    **Pagination:**
    - Use `page` and `limit` to paginate through results
    - Response includes `total_matching` (total products) and `total_pages`
    
    **Example queries:**
    - "casual blue jeans for summer"
    - "formal business attire"
    - "sports shoes for running"
    - "ethnic dress for party"
    """,
)
async def get_recommendations(
    q: str = Query(
        ...,
        description="Search query describing the desired outfit or occasion",
        min_length=1,
        example="casual blue jeans",
    ),
    page: int = Query(
        default=1,
        description="Page number (1-indexed)",
        ge=1,
    ),
    limit: int = Query(
        default=12,
        description="Number of items per page",
        ge=1,
        le=50,
    ),
    top_k: Optional[int] = Query(
        default=None,
        description="[DEPRECATED] Use 'limit' instead. If provided, overrides 'limit' for backward compatibility.",
        ge=1,
        le=50,
    ),
) -> RecommendationResponse:
    """
    Retrieve outfit recommendations for a given text query with pagination.

    Args:
        q: Text description of desired outfit or occasion
        page: Page number (1-indexed, default=1)
        limit: Number of items per page (1-50, default=12)
        top_k: [DEPRECATED] Use 'limit' instead

    Returns:
        RecommendationResponse with ranked product recommendations and pagination info

    Raises:
        HTTPException: If an error occurs during recommendation generation
    """
    try:
        # Handle backward compatibility: top_k overrides limit if provided
        effective_limit = top_k if top_k is not None else limit
        
        # Get recommendations from service with pagination
        result = recommendation_service.get_recommendations(
            query=q, 
            page=page, 
            limit=effective_limit
        )

        return RecommendationResponse(**result)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )

