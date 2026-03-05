import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# --- 1. CHARGEMENT ET ALIGNEMENT ---
print("--- 1. CHARGEMENT ET ALIGNEMENT ---")
df_custom = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features_V2.csv", index_col="track_id")
df_default = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features.csv", header=[0, 1, 2], index_col=0, low_memory=False)

# Alignement strict sur les IDs communs
common_tracks = df_custom.index.intersection(df_default.index)
df_custom = df_custom.loc[common_tracks].copy()
df_default = df_default.loc[common_tracks].copy()

# Encodage de la cible (genre_top) et définition des groupes (artistes)
le = LabelEncoder()
y = le.fit_transform(df_custom["genre_top"])
groups = df_custom["artist_name"].values

# --- 2. NETTOYAGE DES DONNÉES (SUPPRESSION DU TEXTE) ---
print("--- 2. NETTOYAGE DES DONNÉES ---")
cols_to_drop = ['genre_top', 'genres', 'genres_decoded', 'n_subgenres', 'mismatch', 'artist_name', 'track_title']
X_custom = df_custom.drop(columns=[c for c in cols_to_drop if c in df_custom.columns])

# Conversion forcée en numérique pour XGBoost
X_custom = X_custom.apply(pd.to_numeric, errors='coerce').fillna(0)
X_custom = X_custom.select_dtypes(include=[np.number])

X_default = df_default.apply(pd.to_numeric, errors='coerce').fillna(0)
X_default = X_default.select_dtypes(include=[np.number])

# --- 3. SPLIT GROUPÉ PAR ARTISTE ---
print("--- 3. SPLIT GROUPÉ PAR ARTISTE ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_custom, y, groups=groups))

X_train_custom, X_test_custom = X_custom.iloc[train_idx], X_custom.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"✅ Données prêtes : {len(common_tracks)} pistes")

# --- 4. PHASE D'OPTIMISATION (DATA SCIENCE) ---
print("\n--- 4. OPTIMISATION DU MODÈLE XGBOOST (V3) ---")

# Calcul des poids pour compenser le déséquilibre des classes (ex: genre Pop)
poids_entrainement = compute_sample_weight(class_weight='balanced', y=y_train)

# Paramètres optimisés : plus d'arbres, apprentissage plus lent, moins de surapprentissage
xgb_params_v3 = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1
}

print("Entraînement du modèle optimisé en cours...")
start = time.time()
model_v3 = XGBClassifier(**xgb_params_v3)
model_v3.fit(X_train_custom, y_train, sample_weight=poids_entrainement)
print(f"-> Entraînement terminé en {time.time() - start:.2f} secondes")

# --- 5. ÉVALUATION FINALE ---
print("\n--- 5. ÉVALUATION DES PERFORMANCES ---")
preds_v3 = model_v3.predict(X_test_custom)
f1_v3 = f1_score(y_test, preds_v3, average="macro")

print(f"🌟 NOUVEAU F1 MACRO (V3) : {f1_v3:.4f}")
print("\n--- RAPPORT DÉTAILLÉ PAR GENRE ---")
print(classification_report(y_test, preds_v3, target_names=le.classes_))

# --- 6. IMPORTANCE DES FEATURES ---
importances = model_v3.feature_importances_
df_imp = pd.DataFrame({"Feature": X_train_custom.columns, "Importance": importances})
print("\nTop 10 des variables les plus influentes :")
print(df_imp.sort_values(by="Importance", ascending=False).head(10).to_string(index=False))