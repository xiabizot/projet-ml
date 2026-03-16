"""
agent.py — Logique metier Agent IA Classification de Genre Musical
===================================================================
Fonctions de prediction, SHAP, recommandation, Claude API.
Aucune dependance Streamlit.
"""

import re
import os
import warnings
from pathlib import Path
from html import unescape
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env', override=True)

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


# =============================================
# Architecture CNN (identique NB4 — source unique)
# =============================================
class AudioCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# =============================================
# CONFIG
# =============================================
BASE = Path(__file__).parent.parent  # src/ -> PROJET/
AGENT_DIR = BASE / 'outputs' / 'agent'
CNN_DIR = BASE / 'outputs' / 'cnn'
FEATURES_CSV = BASE / 'outputs' / 'features' / 'features_V2.csv'
SPECTRO_DIR = BASE / 'spectrogrammes'
CURATION_DIR = BASE / 'outputs' / 'curation'
META_DIR = BASE / 'data' / 'raw' / 'fma_metadata'

SR = 22050
DURATION = 30

GENRE_EMOJI = {
    'Hip-Hop': '♪', 'Pop': '○', 'Folk': '♩', 'Experimental': '◇',
    'Rock': '♯', 'International': '◎', 'Electronic': '◈', 'Instrumental': '♬',
}

GENRE_SVG = {
    'Hip-Hop': '<svg viewBox="0 0 40 40" width="28" height="28"><circle cx="20" cy="12" r="5" fill="#fafafa"/><rect x="15" y="17" width="10" height="12" rx="3" fill="#fafafa"/><line x1="18" y1="8" x2="14" y2="3" stroke="#fafafa" stroke-width="2"/><line x1="22" y1="8" x2="26" y2="3" stroke="#fafafa" stroke-width="2"/><rect x="12" y="29" width="5" height="8" rx="2" fill="#fafafa"/><rect x="23" y="29" width="5" height="8" rx="2" fill="#fafafa"/></svg>',
    'Rock': '<svg viewBox="0 0 40 40" width="28" height="28"><rect x="17" y="2" width="6" height="30" rx="2" fill="#fafafa"/><ellipse cx="14" cy="34" rx="8" ry="5" fill="none" stroke="#fafafa" stroke-width="2"/><line x1="23" y1="8" x2="30" y2="8" stroke="#fafafa" stroke-width="2"/><line x1="23" y1="14" x2="30" y2="14" stroke="#fafafa" stroke-width="2"/><line x1="23" y1="20" x2="28" y2="20" stroke="#fafafa" stroke-width="2"/></svg>',
    'Folk': '<svg viewBox="0 0 40 40" width="28" height="28"><ellipse cx="15" cy="32" rx="9" ry="6" fill="none" stroke="#fafafa" stroke-width="2"/><line x1="24" y1="32" x2="24" y2="5" stroke="#fafafa" stroke-width="2"/><line x1="24" y1="5" x2="32" y2="8" stroke="#fafafa" stroke-width="2"/><line x1="24" y1="10" x2="32" y2="13" stroke="#fafafa" stroke-width="2"/></svg>',
    'Electronic': '<svg viewBox="0 0 40 40" width="28" height="28"><rect x="5" y="15" width="30" height="18" rx="3" fill="none" stroke="#fafafa" stroke-width="2"/><circle cx="14" cy="24" r="4" fill="none" stroke="#fafafa" stroke-width="1.5"/><circle cx="26" cy="24" r="4" fill="none" stroke="#fafafa" stroke-width="1.5"/><line x1="14" y1="24" x2="14" y2="20" stroke="#fafafa" stroke-width="1.5"/><line x1="26" y1="24" x2="26" y2="20" stroke="#fafafa" stroke-width="1.5"/><rect x="10" y="8" width="20" height="4" rx="1" fill="#fafafa"/></svg>',
    'Experimental': '<svg viewBox="0 0 40 40" width="28" height="28"><polygon points="20,3 35,35 5,35" fill="none" stroke="#fafafa" stroke-width="2"/><circle cx="20" cy="25" r="5" fill="none" stroke="#fafafa" stroke-width="1.5"/><line x1="20" y1="12" x2="20" y2="20" stroke="#fafafa" stroke-width="1.5"/></svg>',
    'International': '<svg viewBox="0 0 40 40" width="28" height="28"><circle cx="20" cy="20" r="15" fill="none" stroke="#fafafa" stroke-width="2"/><ellipse cx="20" cy="20" rx="8" ry="15" fill="none" stroke="#fafafa" stroke-width="1.5"/><line x1="5" y1="20" x2="35" y2="20" stroke="#fafafa" stroke-width="1.5"/><line x1="8" y1="12" x2="32" y2="12" stroke="#fafafa" stroke-width="1"/><line x1="8" y1="28" x2="32" y2="28" stroke="#fafafa" stroke-width="1"/></svg>',
    'Pop': '<svg viewBox="0 0 40 40" width="28" height="28"><circle cx="20" cy="20" r="12" fill="none" stroke="#fafafa" stroke-width="2"/><circle cx="20" cy="20" r="4" fill="#fafafa"/><line x1="20" y1="8" x2="20" y2="2" stroke="#fafafa" stroke-width="2"/><line x1="20" y1="32" x2="20" y2="38" stroke="#fafafa" stroke-width="2"/><line x1="8" y1="20" x2="2" y2="20" stroke="#fafafa" stroke-width="2"/><line x1="32" y1="20" x2="38" y2="20" stroke="#fafafa" stroke-width="2"/></svg>',
    'Instrumental': '<svg viewBox="0 0 40 40" width="28" height="28"><path d="M12,35 Q8,30 10,20 Q12,10 20,5 Q28,10 30,20 Q32,30 28,35 Z" fill="none" stroke="#fafafa" stroke-width="2"/><line x1="16" y1="15" x2="16" y2="30" stroke="#fafafa" stroke-width="1"/><line x1="20" y1="12" x2="20" y2="30" stroke="#fafafa" stroke-width="1"/><line x1="24" y1="15" x2="24" y2="30" stroke="#fafafa" stroke-width="1"/></svg>',
}

FEATURE_TRANSLATION = {
    'spectral_centroid': 'Brillance (centre de gravite frequentiel)',
    'spectral_bandwidth': 'Largeur spectrale (richesse harmonique)',
    'spectral_rolloff': 'Frequence de coupure haute',
    'spectral_contrast': 'Texture sonore (relief spectral)',
    'mfcc': 'Timbre (empreinte vocale/instrumentale)',
    'chroma_stft': 'Contenu harmonique (profil tonal)',
    'tonnetz': 'Relations tonales (accords, harmonie)',
    'rmse': 'Energie / Volume',
    'zcr': 'Bruit haute frequence (distorsion, percussions)',
    'tempo': 'Tempo (BPM)',
}

STAT_TRANSLATION = {
    'mean': 'moyenne', 'std': 'variabilite', 'min': 'minimum',
    'max': 'maximum', 'median': 'mediane', 'skew': 'asymetrie',
    'kurtosis': 'concentration',
}


def translate_feature(feat_name):
    """Traduit un nom de feature technique en description lisible."""
    for key, desc in FEATURE_TRANSLATION.items():
        if feat_name.startswith(key):
            parts = feat_name.split('_')
            stat = parts[-1]
            stat_fr = STAT_TRANSLATION.get(stat, stat)
            return f'{desc} ({stat_fr})'
    return feat_name


# =============================================
# CHARGEMENT ARTEFACTS
# =============================================
_cache = {}


def load_models():
    """Charge tous les artefacts NB9. Retourne un dict."""
    if _cache.get('loaded'):
        return _cache

    _cache['pipe_v1'] = joblib.load(AGENT_DIR / 'pipe_xgb_audio.joblib')
    _cache['lr_nlp'] = joblib.load(AGENT_DIR / 'lr_nlp.joblib')
    _cache['tfidf'] = joblib.load(AGENT_DIR / 'tfidf_vectorizer.joblib')
    _cache['le'] = joblib.load(AGENT_DIR / 'label_encoder.joblib')
    _cache['preprocessor'] = joblib.load(AGENT_DIR / 'preprocessor_v1.joblib')
    _cache['feature_cols'] = list(np.load(AGENT_DIR / 'feature_cols.npy', allow_pickle=True))

    w_path = AGENT_DIR / 'weights_v2.npy'
    if not w_path.exists():
        w_path = AGENT_DIR / 'weights.npy'
    weights = np.load(w_path)
    _cache['w_audio'] = float(weights[0])
    _cache['w_nlp'] = float(weights[1])
    _cache['w_panns'] = float(weights[2]) if len(weights) > 2 else 0.0

    panns_path = AGENT_DIR / 'panns_xgb.joblib'
    _cache['panns_xgb'] = joblib.load(panns_path) if panns_path.exists() else None

    _cache['genres'] = list(_cache['le'].classes_)

    # Data
    df = pd.read_csv(FEATURES_CSV)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed:')]
    _cache['df_feat'] = df

    _cache['embeddings_test'] = np.load(CNN_DIR / 'embeddings_test.npy')
    _cache['track_ids_cnn'] = np.load(CNN_DIR / 'test_track_ids.npy')
    _cache['genres_cnn'] = np.load(CNN_DIR / 'test_genres.npy', allow_pickle=True)
    _cache['emb_map'] = dict(zip(_cache['track_ids_cnn'], range(len(_cache['track_ids_cnn']))))

    suspects_path = CURATION_DIR / 'suspects_top30.csv'
    _cache['df_suspects'] = pd.read_csv(suspects_path) if suspects_path.exists() else pd.DataFrame()

    # PANNs embeddings
    panns_pkl = BASE / 'outputs' / 'transfer_learning' / 'embeddings_panns.pkl'
    _cache['panns_dict'] = {}
    if panns_pkl.exists():
        import pickle
        with open(panns_pkl, 'rb') as f:
            _cache['panns_dict'] = pickle.load(f)

    _cache['loaded'] = True
    return _cache


def load_features_data():
    """Charge le dataframe features avec texte NLP."""
    m = load_models()
    df = m['df_feat'].copy()

    # Ajouter texte NLP si pas deja present
    if 'text' not in df.columns:
        try:
            tracks = pd.read_csv(META_DIR / 'tracks.csv', index_col=0, header=[0, 1])
            small = tracks[tracks[('set', 'subset')] == 'small'].copy()
            feat_ids = set(df['track_id'].values)
            small = small[small.index.isin(feat_ids)]
            text_series = (
                small[('track', 'title')].fillna('').astype(str) + ' ' +
                small[('album', 'title')].fillna('').astype(str) + ' ' +
                small[('artist', 'bio')].fillna('').apply(_clean_html)
            )
            text_map = dict(zip(small.index, text_series.values))
            df['text'] = df['track_id'].map(text_map).fillna('')
        except Exception:
            df['text'] = ''

    return df


def _clean_html(text):
    if pd.isna(text) or text == '':
        return ''
    text = unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# =============================================
# EXTRACTION FEATURES AUDIO
# =============================================
def extract_features_from_audio(y_audio, sr=SR):
    """Extrait les 351 features librosa depuis un signal audio."""
    import librosa
    m = load_models()
    feat_cols = m['feature_cols']
    features = {}

    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
    for i, coef in enumerate(mfcc):
        s = pd.Series(coef)
        for stat, v in zip(
            ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'],
            [np.mean(coef), np.std(coef), np.min(coef), np.max(coef),
             np.median(coef), float(s.skew()), float(s.kurtosis())]
        ):
            features[f'mfcc_{i+1:02d}_{stat}'] = v

    for name, func in [
        ('centroid', librosa.feature.spectral_centroid),
        ('bandwidth', librosa.feature.spectral_bandwidth),
        ('rolloff', librosa.feature.spectral_rolloff),
    ]:
        vals = func(y=y_audio, sr=sr)[0]
        s = pd.Series(vals)
        for stat, v in zip(
            ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'],
            [np.mean(vals), np.std(vals), np.min(vals), np.max(vals),
             np.median(vals), float(s.skew()), float(s.kurtosis())]
        ):
            features[f'spectral_{name}_01_{stat}'] = v

    contrast = librosa.feature.spectral_contrast(y=y_audio, sr=sr)
    for i, band in enumerate(contrast):
        s = pd.Series(band)
        for stat, v in zip(
            ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'],
            [np.mean(band), np.std(band), np.min(band), np.max(band),
             np.median(band), float(s.skew()), float(s.kurtosis())]
        ):
            features[f'spectral_contrast_{i+1:02d}_{stat}'] = v

    chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
    for i, ch in enumerate(chroma):
        s = pd.Series(ch)
        for stat, v in zip(
            ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'],
            [np.mean(ch), np.std(ch), np.min(ch), np.max(ch),
             np.median(ch), float(s.skew()), float(s.kurtosis())]
        ):
            features[f'chroma_stft_{i+1:02d}_{stat}'] = v

    try:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y_audio), sr=sr)
        for i, t in enumerate(tonnetz):
            s = pd.Series(t)
            for stat, v in zip(
                ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'],
                [np.mean(t), np.std(t), np.min(t), np.max(t),
                 np.median(t), float(s.skew()), float(s.kurtosis())]
            ):
                features[f'tonnetz_{i+1:02d}_{stat}'] = v
    except Exception:
        for i in range(6):
            for stat in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']:
                features[f'tonnetz_{i+1:02d}_{stat}'] = 0.0

    for name, vals in [
        ('rmse', librosa.feature.rms(y=y_audio)[0]),
        ('zcr', librosa.feature.zero_crossing_rate(y_audio)[0])
    ]:
        s = pd.Series(vals)
        for stat, v in zip(
            ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'],
            [np.mean(vals), np.std(vals), np.min(vals), np.max(vals),
             np.median(vals), float(s.skew()), float(s.kurtosis())]
        ):
            features[f'{name}_01_{stat}'] = v

    tempo, _ = librosa.beat.beat_track(y=y_audio, sr=sr)
    features['tempo'] = float(tempo)

    feat_df = pd.DataFrame([features])
    for col in feat_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    return feat_df[feat_cols].values


# =============================================
# SHAP
# =============================================
def get_shap_features(feat_vec, pred_enc=None, n=10):
    """Top N features SHAP pour la prediction."""
    try:
        import shap
        m = load_models()
        X_proc = m['preprocessor'].transform(feat_vec)
        explainer = shap.TreeExplainer(m['pipe_v1'].named_steps['clf'])
        sv = explainer.shap_values(X_proc)

        if pred_enc is None:
            pred_enc = m['pipe_v1'].predict(feat_vec)[0]

        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            shap_vals = sv[0, :, pred_enc]
        else:
            shap_vals = sv[pred_enc][0]

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:n]
        return [(m['feature_cols'][j], round(float(shap_vals[j]), 4)) for j in top_idx]
    except Exception:
        return []


# =============================================
# RECOMMANDATION
# =============================================
def _extract_cnn_embedding_from_spec(spec_path):
    """Extrait l'embedding CNN 256D depuis un spectrogramme .npy existant."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        m = load_models()
        spec = np.load(spec_path)
        target_frames = 1291
        if spec.shape[1] >= target_frames:
            spec = spec[:, :target_frames]
        else:
            spec = np.pad(spec, ((0, 0), (0, target_frames - spec.shape[1])))
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        # Reutiliser le meme code CNN
        return _run_cnn_extractor(spec, m)
    except Exception:
        return None


def _run_cnn_extractor(spec_norm, m):
    """Passe un spectrogramme normalise dans le CNN et retourne l'embedding."""

    class EmbExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.conv1, self.bn1 = model.conv1, model.bn1
            self.conv2, self.bn2 = model.conv2, model.bn2
            self.conv3, self.bn3 = model.conv3, model.bn3
            self.conv4, self.bn4 = model.conv4, model.bn4
            self.pool, self.adaptive_pool = model.pool, model.adaptive_pool
            self.fc1, self.fc2, self.dropout = model.fc1, model.fc2, model.dropout

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.adaptive_pool(F.relu(self.bn4(self.conv4(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = CNN_DIR / 'best_audio_cnn.pth'
    if not model_path.exists():
        return None

    cnn = AudioCNN(num_classes=len(m['genres'])).to(device)
    cnn.load_state_dict(torch.load(model_path, map_location=device))
    cnn.eval()

    extractor = EmbExtractor(cnn).to(device)
    extractor.eval()

    tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = extractor(tensor).cpu().numpy()
    return embedding


def _extract_cnn_embedding(mp3_path):
    """Genere le spectrogramme log-mel depuis un MP3 et extrait l'embedding CNN 256D."""
    try:
        import librosa
        m = load_models()

        y_audio, _ = librosa.load(str(mp3_path), sr=SR, mono=True)
        target = SR * DURATION
        if len(y_audio) >= target:
            start = (len(y_audio) - target) // 2
            y_audio = y_audio[start:start + target]
        else:
            y_audio = np.pad(y_audio, (0, target - len(y_audio)))

        S = librosa.feature.melspectrogram(y=y_audio, sr=SR, n_fft=2048,
                                           hop_length=512, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        target_frames = 1291
        if S_db.shape[1] >= target_frames:
            S_db = S_db[:, :target_frames]
        else:
            S_db = np.pad(S_db, ((0, 0), (0, target_frames - S_db.shape[1])))

        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
        return _run_cnn_extractor(S_db, m)
    except Exception:
        return None


def get_recommendations(track_id=None, mp3_path=None, n=5):
    """Top N pistes similaires par cosine similarity CNN."""
    m = load_models()
    embedding = None

    # Piste FMA : utiliser embedding pre-calcule
    if track_id is not None and track_id in m['emb_map']:
        idx = m['emb_map'][track_id]
        embedding = m['embeddings_test'][idx].reshape(1, -1)

    # Piste FMA sans embedding : generer depuis spectrogramme
    elif track_id is not None:
        spec_path = SPECTRO_DIR / f'{int(track_id):06d}.npy'
        if spec_path.exists():
            embedding = _extract_cnn_embedding_from_spec(spec_path)

    # MP3 externe : extraire embedding live
    elif mp3_path is not None:
        embedding = _extract_cnn_embedding(mp3_path)

    if embedding is None:
        return []

    sims = cosine_similarity(embedding, m['embeddings_test'])[0]
    top_n = sims.argsort()[::-1][:n]
    # Si c'est une piste FMA, skip la piste elle-meme
    if track_id is not None and track_id in m['emb_map']:
        top_n = sims.argsort()[::-1][1:n + 1]

    reco = []
    df = m['df_feat']
    for ri in top_n:
        tid = int(m['track_ids_cnn'][ri])
        meta = df[df['track_id'] == tid]
        title = meta['track_title'].values[0] if len(meta) > 0 and 'track_title' in meta.columns else str(tid)
        artist = meta['artist_name'].values[0] if len(meta) > 0 else '?'
        reco.append({
            'track_id': tid,
            'title': str(title),
            'artist': str(artist),
            'genre': str(m['genres_cnn'][ri]),
            'similarity': round(float(sims[ri]) * 100, 1),
        })
    return reco


# =============================================
# CLAUDE API
# =============================================
def explain_with_claude(result, mode='V1'):
    """Explication Claude API. Retourne un str."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        topk_str = ', '.join(
            f"{g} ({p:.1%})" for g, p in result.get('top_k', [])[:3]
        )

        if mode == 'V1':
            prompt = (
                f'Tu es un assistant de classification musicale.\n'
                f'XGBoost a analyse "{result["title"]}" de {result["artist"]}.\n'
                f'Genre predit : {result["pred_genre"]} ({result["confidence"]:.1%})\n'
                f'Top-3 : {topk_str}\n'
                f'Genre reel : {result.get("true_genre", "?")}\n'
                f'Mismatch : {result.get("mismatch", False)}\n'
                f'En 3-4 phrases sobres, explique ce resultat pour un utilisateur streaming.'
            )
            max_tokens = 300
        else:
            shap_str = ''
            if result.get('shap_features'):
                lines = [f'  - {f}: {v:+.4f}' for f, v in result['shap_features'][:5]]
                shap_str = 'SHAP top features :\n' + '\n'.join(lines)

            bias_note = ''
            if 'Blue Dot' in str(result.get('artist', '')):
                bias_note = 'Blue Dot Sessions : sur-represente dans Instrumental.'

            prompt = (
                f'Tu es un expert MIR specialise FMA Small.\n'
                f'"{result["title"]}" - {result["artist"]}\n'
                f'Genre predit : {result["pred_genre"]} ({result["confidence"]:.1%})\n'
                f'Top-3 : {topk_str}\n'
                f'Genre reel : {result.get("true_genre", "?")}\n'
                f'Mismatch : {result.get("mismatch", False)}\n'
                f'Suspect (atypique) : {result.get("suspect", False)}\n'
                f'{shap_str}\n{bias_note}\n'
                f'En 5-6 phrases expertes : 1) ce que le modele detecte acoustiquement '
                f'2) pourquoi une erreur est comprehensible 3) recommandation streaming.'
            )
            max_tokens = 500

        resp = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=max_tokens,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return resp.content[0].text
    except Exception as e:
        return f'Claude API non disponible : {e}'


# =============================================
# RUN AGENT
# =============================================
LABEL_COLS = [
    'track_id', 'track_id_int', 'genre_top', 'artist_name', 'genres_decoded',
    'genres', 'n_subgenres', 'mismatch', 'mismatch_calc', 'track_title',
    'year', 'duration', 'bit_rate', 'text'
]


def run_agent(track_id=None, mp3_path=None, mode='V1', with_claude=False):
    """
    Point d'entree unique de l'agent IA.

    Args:
        track_id: int — ID piste FMA (prioritaire si fourni)
        mp3_path: str/Path — chemin vers un fichier MP3 (si pas de track_id)
        mode: 'V1' ou 'V2'
        with_claude: bool — generer explication Claude API

    Returns:
        dict avec prediction, confiance, top_k, shap, reco, etc.
    """
    m = load_models()
    df = load_features_data()
    fc = [c for c in m['feature_cols'] if c in df.columns]

    result = {
        'track_id': track_id,
        'title': '?',
        'artist': '?',
        'true_genre': '?',
        'pred_genre': None,
        'confidence': 0.0,
        'top_k': [],
        'mismatch': False,
        'mode': mode,
        'pred_audio': None,
        'pred_nlp': None,
        'pred_panns': None,
        'proba_audio': None,
        'proba_nlp': None,
        'proba_v2': None,
    }

    # --- Obtenir les features ---
    text = ''
    if track_id is not None:
        row = df[df['track_id'] == track_id]
        if row.empty:
            result['error'] = f'Track ID {track_id} introuvable'
            return result
        row = row.iloc[0]
        feat_vec = row[fc].values.reshape(1, -1)
        result['title'] = str(row.get('track_title', '?'))
        result['artist'] = str(row.get('artist_name', '?'))
        result['true_genre'] = str(row['genre_top'])
        result['mismatch'] = bool(row.get('mismatch', False))
        text = str(row.get('text', ''))
        if text == 'nan':
            text = ''
    elif mp3_path is not None:
        import librosa
        y_audio, _ = librosa.load(str(mp3_path), sr=SR, mono=True)
        target = SR * DURATION
        if len(y_audio) >= target:
            start = (len(y_audio) - target) // 2
            y_audio = y_audio[start:start + target]
        else:
            y_audio = np.pad(y_audio, (0, target - len(y_audio)))
        feat_vec = extract_features_from_audio(y_audio, SR)
        result['title'] = Path(mp3_path).stem
        result['artist'] = 'Fichier uploade'
    else:
        result['error'] = 'Fournir track_id ou mp3_path'
        return result

    # --- V1 : XGBoost audio ---
    proba_audio = m['pipe_v1'].predict_proba(feat_vec)[0]
    pred_v1_enc = proba_audio.argmax()
    result['pred_audio'] = m['le'].inverse_transform([pred_v1_enc])[0]
    result['proba_audio'] = proba_audio

    if mode == 'V1':
        result['pred_genre'] = result['pred_audio']
        result['confidence'] = float(proba_audio.max())
        result['top_k'] = [
            (m['le'].inverse_transform([i])[0], float(proba_audio[i]))
            for i in proba_audio.argsort()[::-1][:3]
        ]

    # --- V2 : Ensemble ---
    if mode == 'V2':
        proba_nlp = m['lr_nlp'].predict_proba(m['tfidf'].transform([text]))[0]
        result['pred_nlp'] = m['le'].inverse_transform([proba_nlp.argmax()])[0]
        result['proba_nlp'] = proba_nlp

        # PANNs
        has_panns = (m['panns_xgb'] is not None and track_id is not None
                     and track_id in m['panns_dict'])

        if has_panns:
            X_panns = np.array(m['panns_dict'][track_id]).reshape(1, -1)
            proba_panns = m['panns_xgb'].predict_proba(X_panns)[0]
            proba_ensemble = (m['w_audio'] * proba_audio
                              + m['w_nlp'] * proba_nlp
                              + m['w_panns'] * proba_panns)
            result['pred_panns'] = m['le'].inverse_transform([proba_panns.argmax()])[0]
        else:
            # Renormaliser sans PANNs
            w_sum = m['w_audio'] + m['w_nlp']
            proba_ensemble = (m['w_audio'] / w_sum) * proba_audio + (m['w_nlp'] / w_sum) * proba_nlp

        result['proba_v2'] = proba_ensemble
        pred_v2_enc = proba_ensemble.argmax()
        result['pred_genre'] = m['le'].inverse_transform([pred_v2_enc])[0]
        result['confidence'] = float(proba_ensemble.max())
        result['top_k'] = [
            (m['le'].inverse_transform([i])[0], float(proba_ensemble[i]))
            for i in proba_ensemble.argsort()[::-1][:3]
        ]

        # SHAP
        result['shap_features'] = get_shap_features(feat_vec, pred_enc=pred_v2_enc)

        # Suspect
        if not m['df_suspects'].empty and track_id is not None:
            result['suspect'] = int(track_id) in m['df_suspects']['track_id'].values
        else:
            result['suspect'] = False

    # --- Recommandation (V1 et V2) ---
    result['reco'] = get_recommendations(track_id=track_id, mp3_path=mp3_path)

    # --- Claude API ---
    if with_claude:
        result['explication'] = explain_with_claude(result, mode=mode)

    return result
