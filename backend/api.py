# backend/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import numpy as np
import pandas as pd
import joblib
import clip
import torch
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.environ.get('MODEL_DIR', '../model')
DATA_DIR = os.environ.get('DATA_DIR', '../data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

# Load artifacts
print('Loading artifacts...')
clf = joblib.load(os.path.join(MODEL_DIR, 'usage_classifier.joblib'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
image_embeddings = np.load(os.path.join(MODEL_DIR, 'image_embeddings.npy'))
image_ids = pd.read_csv(os.path.join(MODEL_DIR, 'image_ids.csv'))['id'].tolist()

styles_df = pd.read_csv(os.path.join(DATA_DIR, 'styles.csv'), on_bad_lines='skip')
styles_df['id'] = styles_df['id'].astype(int)
styles_df = styles_df.set_index('id')

# CLIP model for text encoding
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()
print('Artifacts loaded. Device:', device)

app = FastAPI(title='Outfit Recommender API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# Mount static images
if os.path.exists(IMAGES_DIR):
    app.mount('/images', StaticFiles(directory=IMAGES_DIR), name='images')


@app.get('/recommend')
def recommend(q: str):
    """Recommend items for a text query."""

    try:
        # Encode text
        tokens = clip.tokenize([q]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(tokens).cpu().numpy()[0]
            text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-10)

        # Predict usage
        try:
            usage_pred = clf.predict(text_emb.reshape(1, -1))[0]
            usage_label = le.inverse_transform([usage_pred])[0]
        except Exception:
            usage_label = None

        # Filter candidates by usage
        candidates_idx = list(range(len(image_ids)))
        if usage_label is not None:
            candidates_idx = [
                i for i, pid in enumerate(image_ids)
                if str(styles_df.loc[pid].get('usage', '')).strip().title()
                == str(usage_label).strip().title()
            ]

        # Fallback if nothing matches
        if len(candidates_idx) == 0:
            candidates_idx = list(range(len(image_ids)))

        # Compute similarities
        candidates_emb = image_embeddings[candidates_idx]
        sims = cosine_similarity(text_emb.reshape(1, -1), candidates_emb)[0]

        top_k = min(12, len(sims))
        top_idx_local = sims.argsort()[-top_k:][::-1]

        # Prepare results
        results = []
        for local_idx in top_idx_local:
            global_idx = candidates_idx[local_idx]
            pid = int(image_ids[global_idx])

            meta = styles_df.loc[pid].to_dict() if pid in styles_df.index else {}

            results.append({
                'id': pid,
                'product': meta.get('productDisplayName', ''),
                'color': meta.get('baseColor', ''),
                'usage': meta.get('usage', ''),
                'score': float(sims[local_idx])
            })

        return JSONResponse({'query': q, 'results': results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
def health():
    return {'status': 'ok'}
