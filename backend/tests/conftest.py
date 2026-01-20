"""
Pytest configuration and fixtures for the Outfit Recommender API tests.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "casual blue jeans",
        "formal business attire",
        "sports shoes",
        "summer dress",
    ]
