"""
Health check routes for API monitoring and status.
"""

from fastapi import APIRouter
from models.schemas import HealthResponse
from services import ml_service
from config import settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health status and verify model/data availability",
)
async def health_check() -> HealthResponse:
    """
    Perform a health check on the API.

    Returns:
        HealthResponse with status and availability information
    """
    missing_files = settings.get_missing_paths()

    return HealthResponse(
        status="ok",
        model_loaded=ml_service.is_loaded(),
        data_available=len(missing_files) == 0,
        missing_files=missing_files if missing_files else None,
    )
