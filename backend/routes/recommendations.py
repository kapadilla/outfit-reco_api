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
    Get personalized outfit recommendations based on a text query.
    
    The system uses CLIP embeddings to understand your query and matches it with 
    fashion products from our catalog. It predicts the usage category (e.g., Casual, 
    Formal) and returns the most similar items ranked by relevance.
    
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
    top_k: Optional[int] = Query(
        default=None,
        description=f"Number of recommendations to return (default: {settings.TOP_K_RESULTS})",
        ge=1,
        le=50,
    ),
) -> RecommendationResponse:
    """
    Retrieve outfit recommendations for a given text query.

    Args:
        q: Text description of desired outfit or occasion
        top_k: Optional number of results to return (1-50)

    Returns:
        RecommendationResponse with ranked product recommendations

    Raises:
        HTTPException: If an error occurs during recommendation generation
    """
    try:
        # Get recommendations from service
        result = recommendation_service.get_recommendations(query=q, top_k=top_k)

        return RecommendationResponse(**result)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )
