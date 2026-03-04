import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("--- 1. CHARGEMENT ET ALIGNEMENT ---")
# Chargement
df_custom = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features_V2.csv", index_col="track_id")
df_default = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features.csv", header=[0, 1, 2], index_col=0, low_memory=False)

# Alignement
common_tracks = df_custom.index.intersection(df_default.index)
df_custom = df_custom.loc[common_tracks].copy()
df_default = df_default.loc[common_tracks].copy()

# Encodage Cible et Groupes
le = LabelEncoder()
y = le.fit_transform(df_custom["genre_top"])
groups = df_custom["artist_name"].values

print("--- 2. NETTOYAGE BLINDÉ (SUPPRESSION TEXTE) ---")
# On supprime toutes les colonnes de texte connues de la V2
cols_to_drop = ['genre_top', 'genres', 'genres_decoded', 'n_subgenres', 'mismatch', 'artist_name', 'track_title']
X_custom = df_custom.drop(columns=[c for c in cols_to_drop if c in df_custom.columns])

# On force tout en numérique et on jette ce qui refuse de se convertir (sécurité max)
X_custom = X_custom.apply(pd.to_numeric, errors='coerce').fillna(0)
X_custom = X_custom.select_dtypes(include=[np.number])

X_default = df_default.apply(pd.to_numeric, errors='coerce').fillna(0)
X_default = X_default.select_dtypes(include=[np.number])

print("--- 3. SPLIT GROUPÉ PAR ARTISTE ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_custom, y, groups=groups))

X_train_custom, X_test_custom = X_custom.iloc[train_idx], X_custom.iloc[test_idx]
X_train_default, X_test_default = X_default.iloc[train_idx], X_default.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"✅ Pistes prêtes : {len(common_tracks)}")
print(f"Features Custom V2 : {X_train_custom.shape[1]} colonnes")
print(f"Features Default   : {X_train_default.shape[1]} colonnes\n")

print("--- 4. ENTRAÎNEMENT XGBOOST ---")
xgb_params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1
}

print(f"Entraînement Modèle 1 (Custom V2)...")
start = time.time()
xgb_custom = XGBClassifier(**xgb_params)
xgb_custom.fit(X_train_custom, y_train)
print(f"-> Terminé en {time.time() - start:.2f} secondes\n")

print(f"Entraînement Modèle 2 (Default)...")
start = time.time()
xgb_default = XGBClassifier(**xgb_params)
xgb_default.fit(X_train_default, y_train)
print(f"-> Terminé en {time.time() - start:.2f} secondes\n")

print("--- 5. PRÉDICTIONS SUR LE SET DE TEST ---")
preds_custom = xgb_custom.predict(X_test_custom)
preds_default = xgb_default.predict(X_test_default)

print("✅ TOUT EST TERMINÉ SANS ERREUR !")
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("\n--- 6. ÉVALUATION DES PERFORMANCES ---")

# 1. Calcul du F1 Macro (La métrique prioritaire du projet)
f1_custom = f1_score(y_test, preds_custom, average="macro")
f1_default = f1_score(y_test, preds_default, average="macro")

print(f"🔥 SCORE F1 MACRO - Custom V2 : {f1_custom:.4f} (Très rapide)")
print(f"🐢 SCORE F1 MACRO - Default   : {f1_default:.4f} (Très lent)")

# 2. Rapport détaillé par genre (Uniquement pour votre modèle star : Custom V2)
print("\n--- RAPPORT DÉTAILLÉ PAR GENRE (CUSTOM V2) ---")
print(classification_report(y_test, preds_custom, target_names=le.classes_))

# 3. Affichage visuel de la Matrice de Confusion
print("\nGénération de la matrice de confusion (une fenêtre va s'ouvrir)...")
cm = confusion_matrix(y_test, preds_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
plt.title("Matrice de Confusion - XGBoost (Features Custom V2)")
plt.tight_layout()
plt.show()

print("✅ PIPELINE 100% TERMINÉE !")