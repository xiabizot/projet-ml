import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

# --- 1. CHARGEMENT ---
print("--- 1. CHARGEMENT (PHASE 4 : FEATURE SELECTION) ---")
df_custom = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features_V2.csv", index_col="track_id")

# --- 2. NETTOYAGE ET FILTRAGE MANUEL ---
# On enlève 'year' pour forcer le modèle à écouter uniquement l'audio
cols_to_drop = ['genre_top', 'genres', 'genres_decoded', 'n_subgenres', 'mismatch', 'artist_name', 'track_title', 'year']
X = df_custom.drop(columns=[c for c in cols_to_drop if c in df_custom.columns])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.select_dtypes(include=[np.number])

le = LabelEncoder()
y = le.fit_transform(df_custom["genre_top"])
groups = df_custom["artist_name"].values

# Split initial pour identifier les features importantes
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train_full, X_test_full = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --- 3. SÉLECTION AUTOMATIQUE DES MEILLEURES FEATURES ---
print("Sélection des 150 meilleures features...")
selector_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector_model.fit(X_train_full, y_train)

# On récupère les scores d'importance et on garde le top 150
importances = selector_model.feature_importances_
indices_top = np.argsort(importances)[-150:] # Les 150 indices les plus hauts
best_cols = X.columns[indices_top]

X_train = X_train_full[best_cols]
X_test = X_test_full[best_cols]

# --- 4. ENTRAÎNEMENT DU MODÈLE FINAL (PHASE 4) ---
print(f"Entraînement avec {len(best_cols)} features sélectionnées...")

xgb_params_v4 = {
    "n_estimators": 400,
    "learning_rate": 0.03, # Plus lent pour plus de précision
    "max_depth": 8,        # Un peu plus profond car moins de features
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1
}

start = time.time()
model_v4 = XGBClassifier(**xgb_params_v4)
model_v4.fit(X_train, y_train)
print(f"-> Entraînement terminé en {time.time() - start:.2f} secondes")

# --- 5. ÉVALUATION ---
preds_v4 = model_v4.predict(X_test)
f1_v4 = f1_score(y_test, preds_v4, average="macro")

print(f"\n📊 F1 MACRO V1 (Base) : 0.5311")
print(f"🚀 F1 MACRO V4 (Top 150 Features) : {f1_v4:.4f}")

print("\n--- RAPPORT DÉTAILLÉ V4 ---")
print(classification_report(y_test, preds_v4, target_names=le.classes_))