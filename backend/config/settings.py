"""
Configuration settings for the Outfit Recommender API.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # API Configuration
    API_TITLE: str = "Outfit Recommender API"
    API_DESCRIPTION: str = "Style & Occasion-Based Outfit Recommendation System"
    API_VERSION: str = "1.0.0"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"

    # CORS Configuration
    CORS_ORIGINS: list = ["*"]  # In production, specify allowed origins
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    # Directory Paths
    MODEL_DIR: Path = BASE_DIR / os.getenv("MODEL_DIR", "model")
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    IMAGES_DIR: Path = DATA_DIR / "images"

    # Model Files
    CLASSIFIER_PATH: Path = MODEL_DIR / "usage_classifier.joblib"
    LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.joblib"
    IMAGE_EMBEDDINGS_PATH: Path = MODEL_DIR / "image_embeddings.npy"
    IMAGE_IDS_PATH: Path = MODEL_DIR / "image_ids.csv"

    # Data Files
    STYLES_CSV_PATH: Path = DATA_DIR / "styles.csv"

    # CLIP Model Configuration
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "ViT-B/32")
    DEVICE: str = os.getenv("DEVICE", "cpu")  # 'cuda' or 'cpu'

    # Recommendation Configuration
    TOP_K_RESULTS: int = 12  # Number of recommendations to return

    def validate_paths(self) -> dict:
        """
        Validate that all required files and directories exist.

        Returns:
            dict: Status of each required path
        """
        paths_status = {
            "model_dir": self.MODEL_DIR.exists(),
            "data_dir": self.DATA_DIR.exists(),
            "images_dir": self.IMAGES_DIR.exists(),
            "classifier": self.CLASSIFIER_PATH.exists(),
            "label_encoder": self.LABEL_ENCODER_PATH.exists(),
            "image_embeddings": self.IMAGE_EMBEDDINGS_PATH.exists(),
            "image_ids": self.IMAGE_IDS_PATH.exists(),
            "styles_csv": self.STYLES_CSV_PATH.exists(),
        }
        return paths_status

    def get_missing_paths(self) -> list:
        """
        Get list of missing required files/directories.

        Returns:
            list: Names of missing paths
        """
        status = self.validate_paths()
        return [name for name, exists in status.items() if not exists]


# Create a singleton settings instance
settings = Settings()
