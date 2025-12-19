"""
Dataset download script for Fashion Product Images Dataset from Kaggle.
Downloads and organizes the dataset for the Outfit Recommender API.
"""

import os
import sys
import kagglehub
from pathlib import Path
from dotenv import load_dotenv
import shutil


def setup_dataset():
    """Download and setup the Fashion Product Images Dataset."""

    # Load environment variables
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if not env_path.exists():
        print("❌ Error: .env file not found!")
        print(f"Please create {env_path} with your KAGGLE_TOKEN")
        sys.exit(1)

    load_dotenv(env_path)

    # Set Kaggle token from environment
    kaggle_token = os.getenv("KAGGLE_TOKEN")
    if not kaggle_token or kaggle_token == "your_kaggle_token_here":
        print("❌ Error: KAGGLE_TOKEN not properly set in .env file")
        print("Please add your Kaggle API token to .env")
        sys.exit(1)

    os.environ["KAGGLE_TOKEN"] = kaggle_token

    print("=" * 60)
    print("Fashion Product Images Dataset Download")
    print("=" * 60)
    print("\n📥 Downloading dataset from Kaggle...")
    print("   (This may take several minutes depending on your connection)")

    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download(
            "paramaggarwal/fashion-product-images-dataset"
        )
        print(f"\n✓ Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("Please check your Kaggle token and internet connection.")
        sys.exit(1)

    # Set up paths
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    dataset_path = Path(dataset_path)

    # Copy styles.csv to data directory
    print("\n📋 Setting up metadata...")

    # Find styles.csv recursively
    styles_csv = None
    for csv_file in dataset_path.rglob("styles.csv"):
        styles_csv = csv_file
        break

    if styles_csv and styles_csv.exists():
        dest_csv = data_dir / "styles.csv"
        shutil.copy2(styles_csv, dest_csv)
        print(f"✓ Copied styles.csv to {dest_csv}")

        # Count rows
        import pandas as pd

        df = pd.read_csv(dest_csv, on_bad_lines="skip")
        print(f"  - {len(df)} products in metadata")
    else:
        print("⚠ Warning: styles.csv not found in downloaded dataset")
        print(f"   Searched in: {dataset_path}")

    # Copy images to data/images directory
    print("\n🖼️  Setting up product images...")

    # Find images directory recursively
    images_src = None
    for img_dir in dataset_path.rglob("images"):
        if img_dir.is_dir():
            images_src = img_dir
            break

    images_dest = data_dir / "images"
    images_dest.mkdir(exist_ok=True)

    if images_src and images_src.exists():
        image_files = list(images_src.glob("*.jpg"))
        total_images = len(image_files)

        if total_images > 0:
            print(f"   Copying {total_images} images...")
            print("   (This will take a few minutes)")

            for idx, img_file in enumerate(image_files, 1):
                shutil.copy2(img_file, images_dest / img_file.name)
                if idx % 1000 == 0:
                    print(f"   Progress: {idx}/{total_images} images copied...")

            print(f"✓ Copied {total_images} images to {images_dest}")
        else:
            print("⚠ No image files found in images directory")
    else:
        print("⚠ Warning: images folder not found in downloaded dataset")
        print(f"   Searched in: {dataset_path}")

    print("\n" + "=" * 60)
    print("✓ Dataset Setup Complete!")
    print("=" * 60)
    print(f"\nDataset location:")
    print(f"  - Metadata: {data_dir / 'styles.csv'}")
    print(f"  - Images:   {images_dest}")
    print(f"\nYou can now start the API with:")
    print(f"  cd backend && uvicorn api:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    setup_dataset()
