"""
Model Training Script for Outfit Recommender

This script retrains the ML models using the full dataset:
1. Computes CLIP image embeddings for all products
2. Trains a Logistic Regression classifier to predict usage categories
3. Applies class balancing techniques to handle imbalanced data:
   - SMOTE oversampling to create synthetic minority class samples
   - class_weight='balanced' in Logistic Regression
4. Saves all model artifacts

Usage:
    python scripts/train_models.py [--sample-size N] [--batch-size B]

Options:
    --sample-size N    Limit to N images (default: all images)
    --batch-size B     Process B images at a time (default: 32)
    --force            Overwrite existing embeddings
    --no-smote         Disable SMOTE oversampling (use class weights only)

Requirements:
    pip install imbalanced-learn  (for SMOTE support)
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
STYLES_CSV = DATA_DIR / "styles.csv"
IMAGES_DIR = DATA_DIR / "images"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Outfit Recommender models")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit to N images (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation (default: 32)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling (use class weights only)",
    )
    return parser.parse_args()


def load_and_clean_metadata(sample_size=None):
    """Load and clean the styles.csv metadata."""
    print("\n📂 Loading metadata...")

    if not STYLES_CSV.exists():
        raise FileNotFoundError(f"styles.csv not found at {STYLES_CSV}")

    styles = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
    print(f"   Raw rows: {len(styles)}")

    # Keep only needed columns
    use_cols = [
        "id",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "usage",
        "productDisplayName",
    ]
    present = [c for c in use_cols if c in styles.columns]
    styles = styles[present].copy()

    # Ensure id is int and drop rows missing id
    styles = styles.dropna(subset=["id"])
    styles["id"] = styles["id"].astype(int)

    # Normalize usage label
    if "usage" in styles.columns:
        styles["usage"] = styles["usage"].astype(str).str.strip().str.title()
    else:
        styles["usage"] = "Unknown"

    # Filter to items with images present
    def image_exists(pid):
        return (IMAGES_DIR / f"{pid}.jpg").exists()

    print("   Checking for existing images...")
    styles["has_image"] = styles["id"].apply(image_exists)
    styles = styles[styles["has_image"]].copy()
    styles.reset_index(drop=True, inplace=True)
    print(f"   Rows with images: {len(styles)}")

    # Optionally sample for faster runs
    if sample_size is not None and sample_size < len(styles):
        styles = styles.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"   Using sample size: {sample_size}")

    return styles


def compute_embeddings(styles, batch_size=32, force=False):
    """Compute CLIP image embeddings for all products."""
    import torch
    import clip

    emb_path = MODEL_DIR / "image_embeddings.npy"
    ids_path = MODEL_DIR / "image_ids.csv"

    # Check if embeddings already exist
    if emb_path.exists() and ids_path.exists() and not force:
        existing_ids = pd.read_csv(ids_path)["id"].tolist()
        if len(existing_ids) == len(styles):
            print(
                "\n✓ Embeddings already exist and match dataset size. Use --force to recompute."
            )
            return np.load(emb_path), existing_ids
        else:
            print(
                f"\n⚠ Existing embeddings ({len(existing_ids)}) don't match dataset ({len(styles)}). Recomputing..."
            )

    print("\n🔄 Computing CLIP embeddings...")
    print(f"   This may take a while for {len(styles)} images...")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    image_ids = []
    embeddings = []
    failed = []

    product_ids = styles["id"].tolist()
    start_time = time.time()
    total_batches = (len(product_ids) + batch_size - 1) // batch_size

    print(f"   Total images: {len(product_ids)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {total_batches}")
    print()

    # Process in batches for efficiency
    for i in tqdm(
        range(0, len(product_ids), batch_size),
        desc="Computing embeddings",
        total=total_batches,
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        batch_ids = product_ids[i : i + batch_size]
        batch_images = []
        valid_ids = []

        for pid in batch_ids:
            img_path = IMAGES_DIR / f"{pid}.jpg"
            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image)
                batch_images.append(img_tensor)
                valid_ids.append(int(pid))
            except Exception as e:
                failed.append((pid, str(e)))
                continue

        if not batch_images:
            continue

        # Stack and encode batch
        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_emb = model.encode_image(batch_tensor)
            batch_emb = batch_emb.cpu().numpy()
            # Normalize each embedding
            norms = np.linalg.norm(batch_emb, axis=1, keepdims=True) + 1e-10
            batch_emb = batch_emb / norms

        embeddings.extend(batch_emb)
        image_ids.extend(valid_ids)

    elapsed = time.time() - start_time
    print(f"\n   Computed {len(image_ids)} embeddings in {elapsed:.1f}s")
    print(f"   Speed: {len(image_ids) / elapsed:.1f} images/second")

    if failed:
        print(f"   ⚠ Failed to process {len(failed)} images")

    # Stack all embeddings
    image_embeddings = np.vstack(embeddings)

    # Save embeddings
    np.save(emb_path, image_embeddings)
    pd.DataFrame({"id": image_ids}).to_csv(ids_path, index=False)
    print(f"\n✓ Saved embeddings: {image_embeddings.shape}")
    print(f"   - {emb_path}")
    print(f"   - {ids_path}")

    return image_embeddings, image_ids


def train_classifier(styles, image_embeddings, image_ids, use_smote=True):
    """Train usage classifier on the embeddings with class balancing.
    
    Args:
        styles: DataFrame with product metadata
        image_embeddings: CLIP embeddings for images
        image_ids: List of product IDs
        use_smote: If True, apply SMOTE oversampling for minority classes
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import joblib

    print("\n🎯 Training usage classifier (with class balancing)...")

    # Build y labels aligned to image_ids
    styles_indexed = styles.set_index("id")
    y = []
    valid_indices = []

    for i, pid in enumerate(image_ids):
        if pid in styles_indexed.index:
            usage = styles_indexed.loc[pid, "usage"]
            if pd.isna(usage):
                usage = "Unknown"
            y.append(str(usage))
            valid_indices.append(i)

    y = pd.Series(y)
    image_embeddings_valid = image_embeddings[valid_indices]

    # Remove classes with only one instance (can't stratify)
    usage_counts = y.value_counts()
    single_instance = usage_counts[usage_counts < 2].index

    if len(single_instance) > 0:
        print(f"   Removing {len(single_instance)} categories with only 1 sample")
        mask = ~y.isin(single_instance)
        y = y[mask].reset_index(drop=True)
        image_embeddings_valid = image_embeddings_valid[mask.values]

    print(f"   Training on {len(y)} samples")
    print(f"   Usage categories: {y.nunique()}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Print class distribution (before balancing)
    print("\n   Class distribution (before balancing):")
    for cls, count in usage_counts.head(10).items():
        print(f"     - {cls}: {count} ({count/len(y)*100:.1f}%)")
    if len(usage_counts) > 10:
        print(f"     ... and {len(usage_counts) - 10} more categories")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        image_embeddings_valid, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    print(f"\n   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Apply SMOTE oversampling if requested
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            
            print("\n   Applying SMOTE oversampling...")
            
            # Determine minimum samples per class for SMOTE
            unique, counts = np.unique(y_train, return_counts=True)
            min_count = min(counts)
            
            # SMOTE needs at least k_neighbors+1 samples per class (default k=5)
            k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
            
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                print(f"   Before SMOTE: {len(X_train)} samples")
                print(f"   After SMOTE:  {len(X_train_balanced)} samples")
                
                # Show new distribution
                unique_new, counts_new = np.unique(y_train_balanced, return_counts=True)
                print("\n   Class distribution (after SMOTE):")
                for cls_idx, count in zip(unique_new, counts_new):
                    cls_name = le.classes_[cls_idx]
                    print(f"     - {cls_name}: {count}")
                
                X_train = X_train_balanced
                y_train = y_train_balanced
            else:
                print("   ⚠ Not enough samples for SMOTE, skipping...")
                
        except ImportError:
            print("   ⚠ imbalanced-learn not installed, skipping SMOTE.")
            print("   Install with: pip install imbalanced-learn")
            print("   Proceeding with class_weight='balanced' only...")

    # Train classifier with balanced class weights
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    clf = LogisticRegression(
        max_iter=2000, 
        n_jobs=-1, 
        verbose=0, 
        random_state=42,
        class_weight='balanced'  # Key change: penalizes minority class misclassification more
    )
    print("\n   Training Logistic Regression with class_weight='balanced'...")

    start_time = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"   Training completed in {elapsed:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   ✓ Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Classification report
    print("\n   Classification Report:")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )
    for line in report.split("\n"):
        print(f"   {line}")

    # Confusion matrix
    print("\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Classes: {le.classes_}")
    print(cm)

    # Save artifacts
    clf_path = MODEL_DIR / "usage_classifier.joblib"
    le_path = MODEL_DIR / "label_encoder.joblib"

    joblib.dump(clf, clf_path)
    joblib.dump(le, le_path)

    print(f"\n✓ Saved classifier artifacts:")
    print(f"   - {clf_path}")
    print(f"   - {le_path}")

    return clf, le


def main():
    args = parse_args()

    print("=" * 60)
    print("  Outfit Recommender - Model Training")
    print("=" * 60)

    # Check prerequisites
    if not DATA_DIR.exists():
        print(f"\n❌ Error: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    if not IMAGES_DIR.exists():
        print(f"\n❌ Error: Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    # Load metadata
    styles = load_and_clean_metadata(sample_size=args.sample_size)

    # Compute embeddings
    image_embeddings, image_ids = compute_embeddings(
        styles, batch_size=args.batch_size, force=args.force
    )

    # Train classifier (with class balancing)
    use_smote = not args.no_smote
    clf, le = train_classifier(styles, image_embeddings, image_ids, use_smote=use_smote)

    # Summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n  📊 Summary:")
    print(f"     - Products processed: {len(image_ids)}")
    print(f"     - Embedding dimensions: {image_embeddings.shape[1]}")
    print(f"     - Usage categories: {len(le.classes_)}")
    print(f"\n  📁 Model artifacts saved to: {MODEL_DIR}")
    print(
        f"     - image_embeddings.npy ({image_embeddings.nbytes / 1024 / 1024:.1f} MB)"
    )
    print(f"     - image_ids.csv")
    print(f"     - usage_classifier.joblib")
    print(f"     - label_encoder.joblib")
    print("\n  🚀 Restart the API server to use the new models!")
    print("=" * 60)


if __name__ == "__main__":
    main()
