"""
ML Model Service - Handles loading and inference with CLIP and classifier models.
"""

import numpy as np
import pandas as pd
import joblib
import clip
import torch
from pathlib import Path
from typing import Tuple, Optional
import logging

from config import settings

logger = logging.getLogger(__name__)


class MLService:
    """Service for managing ML models and performing inference."""

    def __init__(self):
        """Initialize the ML service."""
        self.clf = None
        self.label_encoder = None
        self.image_embeddings = None
        self.image_ids = None
        self.styles_df = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = None
        self._is_loaded = False

    def load_models(self) -> None:
        """
        Load all required ML models and data.

        Raises:
            FileNotFoundError: If required model files are missing
            Exception: If model loading fails
        """
        try:
            logger.info("Loading ML models and artifacts...")

            # Validate paths exist
            missing = settings.get_missing_paths()
            if missing:
                raise FileNotFoundError(
                    f"Missing required files: {', '.join(missing)}. "
                    f"Run 'python scripts/download_dataset.py' to download the dataset."
                )

            # Load classifier and label encoder
            logger.info("Loading usage classifier...")
            self.clf = joblib.load(settings.CLASSIFIER_PATH)

            logger.info("Loading label encoder...")
            self.label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)

            # Load pre-computed image embeddings
            logger.info("Loading image embeddings...")
            self.image_embeddings = np.load(settings.IMAGE_EMBEDDINGS_PATH)

            # Load image IDs
            logger.info("Loading image IDs...")
            image_ids_df = pd.read_csv(settings.IMAGE_IDS_PATH)
            self.image_ids = image_ids_df["id"].tolist()

            # Load styles metadata
            logger.info("Loading product metadata...")
            self.styles_df = pd.read_csv(settings.STYLES_CSV_PATH, on_bad_lines="skip")
            self.styles_df["id"] = self.styles_df["id"].astype(int)
            self.styles_df = self.styles_df.set_index("id")

            # Load CLIP model for text encoding
            logger.info(f"Loading CLIP model: {settings.CLIP_MODEL}")
            self.device = (
                settings.DEVICE
                if torch.cuda.is_available() or settings.DEVICE == "cpu"
                else "cpu"
            )
            self.clip_model, self.clip_preprocess = clip.load(
                settings.CLIP_MODEL, device=self.device
            )
            self.clip_model.eval()

            self._is_loaded = True
            logger.info(f"✓ All models loaded successfully. Device: {self.device}")
            logger.info(f"  - Loaded {len(self.image_ids)} product embeddings")
            logger.info(f"  - Loaded {len(self.styles_df)} product metadata entries")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._is_loaded = False
            raise

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query using CLIP model.

        Args:
            text: Text query to encode

        Returns:
            Normalized text embedding as numpy array
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            # Normalize embedding
            text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-10)

        return text_emb

    def predict_usage(self, text_embedding: np.ndarray) -> Optional[str]:
        """
        Predict usage category from text embedding.

        Args:
            text_embedding: Normalized text embedding

        Returns:
            Predicted usage category label or None if prediction fails
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        try:
            usage_pred = self.clf.predict(text_embedding.reshape(1, -1))[0]
            usage_label = self.label_encoder.inverse_transform([usage_pred])[0]
            return usage_label
        except Exception as e:
            logger.warning(f"Usage prediction failed: {e}")
            return None

    def get_product_metadata(self, product_id: int) -> dict:
        """
        Get product metadata from styles dataframe.

        Args:
            product_id: Product ID

        Returns:
            Dictionary with product metadata
        """
        if product_id in self.styles_df.index:
            return self.styles_df.loc[product_id].to_dict()
        return {}

    @property
    def total_products(self) -> int:
        """Get total number of products."""
        return len(self.image_ids) if self.image_ids else 0


# Create singleton instance
ml_service = MLService()
