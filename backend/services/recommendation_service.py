"""
Recommendation Service - Handles recommendation logic and ranking.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import logging

from services.ml_service import ml_service
from models.schemas import ProductItem
from config import settings

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating outfit recommendations."""

    def __init__(self):
        """Initialize the recommendation service."""
        self.ml_service = ml_service

    def get_recommendations(
        self, query: str, page: int = 1, limit: int = 12
    ) -> Dict[str, any]:
        """
        Get outfit recommendations for a text query with pagination support.

        Args:
            query: Text description of desired outfit
            page: Page number (1-indexed, default=1)
            limit: Number of items per page (default=12, max=50)

        Returns:
            Dictionary containing:
                - query: Original query
                - predicted_usage: Predicted usage category
                - results: List of recommended products for current page
                - total_results: Number of results in current page
                - page: Current page number
                - limit: Items per page
                - total_matching: Total matching products before pagination
                - total_pages: Total number of pages
        """
        if not self.ml_service.is_loaded():
            raise RuntimeError("ML models not loaded")

        # Clamp limit to valid range
        limit = max(1, min(50, limit))
        page = max(1, page)

        # Encode text query using CLIP
        logger.info(f"Processing query: {query}")
        text_emb = self.ml_service.encode_text(query)

        # Predict usage category
        usage_label = self.ml_service.predict_usage(text_emb)
        logger.info(f"Predicted usage: {usage_label}")

        # Sanitize usage label for display (convert Nan to General)
        display_usage = usage_label
        if usage_label is None or str(usage_label).lower() in ["nan", "none", ""]:
            display_usage = "General"

        # Filter candidates by usage category
        candidates_idx = self._filter_by_usage(usage_label)
        
        # Get total matching count for pagination
        total_matching = len(candidates_idx)
        total_pages = max(1, (total_matching + limit - 1) // limit)
        
        # Clamp page to valid range
        page = min(page, total_pages)

        # Compute similarities and rank with pagination
        results = self._rank_candidates_paginated(
            text_emb, candidates_idx, page, limit
        )

        return {
            "query": query,
            "predicted_usage": display_usage,
            "results": results,
            "total_results": len(results),
            "page": page,
            "limit": limit,
            "total_matching": total_matching,
            "total_pages": total_pages,
        }


    def _filter_by_usage(self, usage_label: Optional[str]) -> List[int]:
        """
        Filter product candidates by predicted usage category.

        Args:
            usage_label: Predicted usage category

        Returns:
            List of candidate product indices
        """
        # Start with all products
        candidates_idx = list(range(self.ml_service.total_products))

        # If usage prediction succeeded, filter by usage
        if usage_label is not None:
            filtered_idx = []
            for i, pid in enumerate(self.ml_service.image_ids):
                meta = self.ml_service.get_product_metadata(pid)
                product_usage = str(meta.get("usage", "")).strip().title()

                if product_usage == str(usage_label).strip().title():
                    filtered_idx.append(i)

            # Use filtered results if we found matches
            if len(filtered_idx) > 0:
                candidates_idx = filtered_idx
                logger.info(
                    f"Filtered to {len(candidates_idx)} products with usage: {usage_label}"
                )
            else:
                logger.warning(
                    f"No products found for usage: {usage_label}, using all products"
                )

        return candidates_idx

    def _rank_candidates(
        self, text_emb: np.ndarray, candidates_idx: List[int], top_k: int
    ) -> List[ProductItem]:
        """
        Rank candidates by cosine similarity to text embedding.

        Args:
            text_emb: Text query embedding
            candidates_idx: Indices of candidate products
            top_k: Number of top results to return

        Returns:
            List of ProductItem objects ranked by similarity
        """
        # Get embeddings for candidates
        candidates_emb = self.ml_service.image_embeddings[candidates_idx]

        # Compute cosine similarities
        sims = cosine_similarity(text_emb.reshape(1, -1), candidates_emb)[0]

        # Get top-k indices
        top_k = min(top_k, len(sims))
        top_idx_local = sims.argsort()[-top_k:][::-1]

        # Helper to safely get string value from metadata
        def get_safe_str(val):
            if val is None:
                return ""
            # Check for NaN (pandas often uses float nan for missing values)
            try:
                if np.isnan(val):
                    return ""
            except TypeError:
                pass  # not a number
            return str(val)

        # Build results
        results = []
        for local_idx in top_idx_local:
            global_idx = candidates_idx[local_idx]
            raw_id = self.ml_service.image_ids[global_idx]
            
            # Handle potential NaN or float values from pandas CSV reading
            try:
                if isinstance(raw_id, float) and np.isnan(raw_id):
                    logger.warning(f"Skipping NaN product ID at index {global_idx}")
                    continue
                product_id = int(raw_id)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid product ID at index {global_idx}: {raw_id}, skipping")
                continue

            # Get product metadata
            meta = self.ml_service.get_product_metadata(product_id)

            results.append(
                ProductItem(
                    id=product_id,
                    product=get_safe_str(meta.get("productDisplayName", "")),
                    color=get_safe_str(meta.get("baseColour", "")),
                    usage=get_safe_str(meta.get("usage", "")),
                    score=float(sims[local_idx]),
                    image_url=f"/images/{product_id}.jpg",
                )
            )

        logger.info(f"Returning top {len(results)} recommendations")
        return results

    def _rank_candidates_paginated(
        self, text_emb: np.ndarray, candidates_idx: List[int], page: int, limit: int
    ) -> List[ProductItem]:
        """
        Rank candidates by cosine similarity with pagination support.

        Args:
            text_emb: Text query embedding
            candidates_idx: Indices of candidate products
            page: Page number (1-indexed)
            limit: Number of items per page

        Returns:
            List of ProductItem objects for the requested page
        """
        # Get embeddings for candidates
        candidates_emb = self.ml_service.image_embeddings[candidates_idx]

        # Compute cosine similarities
        sims = cosine_similarity(text_emb.reshape(1, -1), candidates_emb)[0]

        # Sort all candidates by similarity (descending)
        sorted_indices = sims.argsort()[::-1]
        
        # Calculate pagination slice
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_indices = sorted_indices[start_idx:end_idx]

        # Helper to safely get string value from metadata
        def get_safe_str(val):
            if val is None:
                return ""
            try:
                if np.isnan(val):
                    return ""
            except TypeError:
                pass
            return str(val)

        # Build results for current page
        results = []
        for local_idx in page_indices:
            global_idx = candidates_idx[local_idx]
            raw_id = self.ml_service.image_ids[global_idx]

            # Handle potential NaN or float values
            try:
                if isinstance(raw_id, float) and np.isnan(raw_id):
                    logger.warning(f"Skipping NaN product ID at index {global_idx}")
                    continue
                product_id = int(raw_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid product ID at index {global_idx}: {raw_id}, skipping")
                continue

            # Get product metadata
            meta = self.ml_service.get_product_metadata(product_id)

            results.append(
                ProductItem(
                    id=product_id,
                    product=get_safe_str(meta.get("productDisplayName", "")),
                    color=get_safe_str(meta.get("baseColour", "")),
                    usage=get_safe_str(meta.get("usage", "")),
                    score=float(sims[local_idx]),
                    image_url=f"/images/{product_id}.jpg",
                )
            )

        logger.info(f"Returning page {page} with {len(results)} recommendations")
        return results


# Create singleton instance
recommendation_service = RecommendationService()

