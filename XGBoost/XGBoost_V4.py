import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# --- 1. PRÉPARATION DES DONNÉES ---
print("--- 1. CHARGEMENT ET ALIGNEMENT ---")
df_v2 = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features_V2.csv", index_col="track_id")
df_v1 = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features.csv", header=[0, 1, 2], index_col=0, low_memory=False)

# Alignement pour avoir les mêmes pistes dans les deux tests
common_tracks = df_v2.index.intersection(df_v1.index)
df_v2 = df_v2.loc[common_tracks].copy()
df_v1 = df_v1.loc[common_tracks].copy()

# Cible et Groupes (communs aux deux)
le = LabelEncoder()
y = le.fit_transform(df_v2["genre_top"])
groups = df_v2["artist_name"].values

# --- 2. FONCTION DE NETTOYAGE ET SCALING (Audit Xia) ---
def prepare_features(df, is_v1=False):
    # Suppression du texte et colonnes inutiles
    cols_to_drop = ['genre_top', 'genres', 'genres_decoded', 'n_subgenres', 'mismatch', 'artist_name', 'track_title']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Correction des NaNs avec la médiane (Recommandation Xia)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    
    # RobustScaler pour gérer les outliers (Recommandation Xia)
    scaler = RobustScaler()
    return scaler.fit_transform(X)

X_v2_scaled = prepare_features(df_v2)
X_v1_scaled = prepare_features(df_v1, is_v1=True)

# --- 3. SPLIT ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_v2_scaled, y, groups=groups))

y_train, y_test = y[train_idx], y[test_idx]

# --- 4. LE DUEL : XGBOOST V1 VS V2 ---
xgb_params = {
    "n_estimators": 250,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1
}

def train_and_eval(X_scaled, name):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    print(f"\nEntraînement Modèle {name} ({X_train.shape[1]} features)...")
    start = time.time()
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    duration = time.time() - start
    
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average="macro")
    print(f"-> {name} terminé en {duration:.2f}s | F1 Macro : {score:.4f}")
    return score

score_v2 = train_and_eval(X_v2_scaled, "V2 (MAISON)")
score_v1 = train_and_eval(X_v1_scaled, "V1 (FMA)")

print("\n--- BILAN FINAL POUR XIA ---")
print(f"Gagnant au score : {'V2' if score_v2 > score_v1 else 'V1'}")
print(f"Différence de performance : {abs(score_v2 - score_v1):.4f}")