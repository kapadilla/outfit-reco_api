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

    def get_recommendations(self, query: str, top_k: int = None) -> Dict[str, any]:
        """
        Get outfit recommendations for a text query.

        Args:
            query: Text description of desired outfit
            top_k: Number of recommendations to return (default from settings)

        Returns:
            Dictionary containing:
                - query: Original query
                - predicted_usage: Predicted usage category
                - results: List of recommended products
                - total_results: Number of results returned
        """
        if not self.ml_service.is_loaded():
            raise RuntimeError("ML models not loaded")

        if top_k is None:
            top_k = settings.TOP_K_RESULTS

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

        # Compute similarities and rank
        results = self._rank_candidates(text_emb, candidates_idx, top_k)

        return {
            "query": query,
            "predicted_usage": display_usage,
            "results": results,
            "total_results": len(results),
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
            product_id = int(self.ml_service.image_ids[global_idx])

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


# Create singleton instance
recommendation_service = RecommendationService()
