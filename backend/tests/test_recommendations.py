"""
Tests for the /recommend endpoint with pagination support.
"""

import pytest


class TestRecommendEndpointPagination:
    """Tests for pagination functionality in the /recommend endpoint."""

    def test_default_pagination(self, client):
        """Test that default pagination returns page 1 with 12 items."""
        response = client.get("/recommend", params={"q": "casual jeans"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check pagination fields are present
        assert "page" in data
        assert "limit" in data
        assert "total_matching" in data
        assert "total_pages" in data
        
        # Check default values
        assert data["page"] == 1
        assert data["limit"] == 12
        assert data["total_results"] <= 12

    def test_custom_page_and_limit(self, client):
        """Test custom page and limit parameters."""
        response = client.get("/recommend", params={
            "q": "casual jeans",
            "page": 2,
            "limit": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["page"] == 2
        assert data["limit"] == 5
        assert data["total_results"] <= 5

    def test_page_one_vs_page_two_different_results(self, client):
        """Test that page 1 and page 2 return different results."""
        params = {"q": "casual", "limit": 5}
        
        response_page1 = client.get("/recommend", params={**params, "page": 1})
        response_page2 = client.get("/recommend", params={**params, "page": 2})
        
        assert response_page1.status_code == 200
        assert response_page2.status_code == 200
        
        data1 = response_page1.json()
        data2 = response_page2.json()
        
        # If there are results on both pages, they should be different
        if data1["results"] and data2["results"]:
            ids_page1 = {item["id"] for item in data1["results"]}
            ids_page2 = {item["id"] for item in data2["results"]}
            assert ids_page1.isdisjoint(ids_page2), "Page 1 and 2 should have different items"

    def test_total_pages_calculation(self, client):
        """Test that total_pages is calculated correctly."""
        response = client.get("/recommend", params={
            "q": "casual",
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # total_pages should be ceil(total_matching / limit)
        expected_total_pages = max(1, (data["total_matching"] + 9) // 10)
        assert data["total_pages"] == expected_total_pages

    def test_backward_compat_top_k(self, client):
        """Test that top_k parameter still works for backward compatibility."""
        response = client.get("/recommend", params={
            "q": "casual jeans",
            "top_k": 8
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # top_k should override limit
        assert data["limit"] == 8
        assert data["total_results"] <= 8

    def test_top_k_overrides_limit(self, client):
        """Test that top_k takes precedence over limit when both provided."""
        response = client.get("/recommend", params={
            "q": "casual jeans",
            "limit": 5,
            "top_k": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # top_k should override limit
        assert data["limit"] == 10

    def test_page_beyond_results_returns_last_page(self, client):
        """Test requesting a page beyond available results."""
        # First get total pages
        response1 = client.get("/recommend", params={
            "q": "casual",
            "limit": 10
        })
        data1 = response1.json()
        total_pages = data1["total_pages"]
        
        # Request page way beyond
        response2 = client.get("/recommend", params={
            "q": "casual",
            "page": total_pages + 100,
            "limit": 10
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should clamp to last page
        assert data2["page"] <= total_pages

    def test_invalid_page_value(self, client):
        """Test that page value below 1 returns validation error."""
        response = client.get("/recommend", params={
            "q": "casual jeans",
            "page": 0
        })
        
        assert response.status_code == 422  # Validation error

    def test_invalid_limit_value(self, client):
        """Test that limit value outside 1-50 range returns validation error."""
        # Test limit = 0
        response = client.get("/recommend", params={
            "q": "casual jeans",
            "limit": 0
        })
        assert response.status_code == 422
        
        # Test limit > 50
        response = client.get("/recommend", params={
            "q": "casual jeans",
            "limit": 100
        })
        assert response.status_code == 422

    def test_response_has_all_required_fields(self, client):
        """Test that response contains all required fields."""
        response = client.get("/recommend", params={"q": "summer dress"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        required_fields = [
            "query", "predicted_usage", "results", "total_results",
            "page", "limit", "total_matching", "total_pages"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_result_items_have_required_fields(self, client):
        """Test that each result item has required fields."""
        response = client.get("/recommend", params={"q": "casual jeans", "limit": 1})
        
        assert response.status_code == 200
        data = response.json()
        
        if data["results"]:
            item = data["results"][0]
            required_item_fields = ["id", "product", "color", "usage", "score", "image_url"]
            for field in required_item_fields:
                assert field in item, f"Missing required item field: {field}"


class TestRecommendEndpointBasic:
    """Basic tests for /recommend endpoint."""

    def test_health_check(self, client):
        """Test the health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_missing_query_parameter(self, client):
        """Test that missing 'q' parameter returns error."""
        response = client.get("/recommend")
        assert response.status_code == 422  # Validation error

    def test_empty_query_rejected(self, client):
        """Test that empty query string is rejected."""
        response = client.get("/recommend", params={"q": ""})
        assert response.status_code == 422
