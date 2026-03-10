"""
=================================================================================
XGBoost V1 - CONTRAT D'EXPÉRIENCE PARTAGÉ
=================================================================================
PARAMÈTRES FIXES (communs à tous les modèles):
  - Seed              : 42
  - Split             : GroupShuffleSplit 80/20 par artiste
  - Preprocessing     : SimpleImputer(median) dans Pipeline
  - Métrique          : F1 Macro
  - Features          : 351 features audio
  - Validation        : GroupKFold 5-fold
  - Pondération       : compute_sample_weight('balanced')
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =================================================================================
# CONFIGURATION GLOBALE
# =================================================================================
SEED       = 42
TRAIN_SIZE = 0.8
TEST_SIZE  = 0.2
CV_FOLDS   = 5
METRIC     = 'f1_macro'

BASE            = Path.cwd()
FEATURES_V2_CSV = BASE / 'Data' / 'features_V2.csv'
RESULTS_PATH    = BASE / 'XGBoost' / 'results_V1.csv'

print("=" * 90)
print("XGBoost V1 - CONTRAT D'EXPÉRIENCE PARTAGÉ")
print("=" * 90)
print(f"\n📋 PARAMÈTRES:")
print(f"   Seed                   : {SEED}")
print(f"   Split train/test       : {TRAIN_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Stratégie de split     : GroupShuffleSplit par artiste")
print(f"   Validation croisée     : GroupKFold ({CV_FOLDS} folds)")
print(f"   Métrique principale    : {METRIC}")
print(f"   Preprocessing          : SimpleImputer(median) dans Pipeline")
print(f"   Scaling                : Aucun")

np.random.seed(SEED)

# =================================================================================
# ÉTAPE 1: CHARGEMENT DES DONNÉES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
print(f"{'=' * 90}")

start_load = time.time()
df = pd.read_csv(FEATURES_V2_CSV, index_col="track_id")
duration_load = time.time() - start_load

print(f"\n   Fichier        : features_V2.csv")
print(f"   Chargé en      : {duration_load:.2f}s")
print(f"   Dimensions     : {df.shape[0]} pistes × {df.shape[1]} colonnes")

# =================================================================================
# ÉTAPE 2: PRÉPARATION DES DONNÉES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 2: PRÉPARATION DES DONNÉES")
print(f"{'=' * 90}")

COLS_TO_EXCLUDE = [
    'genre_top', 'genres', 'genres_decoded', 'n_subgenres',
    'mismatch', 'artist_name', 'track_title',
    'year', 'duration', 'bit_rate'
]
cols_dropped = [c for c in COLS_TO_EXCLUDE if c in df.columns]
X = df.drop(columns=cols_dropped)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.select_dtypes(include=[np.number])

print(f"\n   Colonnes exclues   : {len(cols_dropped)}")
print(f"   Features audio     : {X.shape[1]} colonnes")
print(f"   Valeurs manquantes : {X.isna().sum().sum()} (imputées dans le Pipeline)")

le = LabelEncoder()
y_enc = le.fit_transform(df["genre_top"])

print(f"\n   Cible encodée (genre_top):")
print(f"   Classes : {list(le.classes_)}")
for cls, idx in zip(le.classes_, range(len(le.classes_))):
    count = (y_enc == idx).sum()
    print(f"      {cls:20s}: {count:5d} pistes ({count/len(y_enc)*100:5.1f}%)")

groups = df["artist_name"].values
unique_artists = len(np.unique(groups))
print(f"\n   Artistes uniques           : {unique_artists}")
print(f"   Pistes par artiste (moy.)  : {len(groups) / unique_artists:.1f}")

# =================================================================================
# ÉTAPE 3: DIVISION TRAIN/TEST
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 3: DIVISION TRAIN/TEST")
print(f"{'=' * 90}")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

X_train_raw  = X.iloc[train_idx]
X_test_raw   = X.iloc[test_idx]
y_train      = y_enc[train_idx]
y_test       = y_enc[test_idx]
groups_train = groups[train_idx]

overlap = len(set(groups[train_idx]) & set(groups[test_idx]))

print(f"\n   Train          : {len(train_idx)} pistes ({len(train_idx)/len(y_enc)*100:.1f}%)")
print(f"   Test           : {len(test_idx)} pistes ({len(test_idx)/len(y_enc)*100:.1f}%)")
print(f"   Artistes communs train/test : {overlap}")

# =================================================================================
# ÉTAPE 4: PIPELINE
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 4: CONSTRUCTION DU PIPELINE")
print(f"{'=' * 90}")

xgb_params = {
    "n_estimators"  : 300,
    "learning_rate" : 0.1,
    "max_depth"     : 6,
    "objective"     : "multi:softprob",
    "num_class"     : len(le.classes_),
    "tree_method"   : "hist",
    "random_state"  : SEED,
    "n_jobs"        : -1,
    "verbosity"     : 0
}

pipe_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf',     XGBClassifier(**xgb_params))
])

print(f"\n   Pipeline       : SimpleImputer(median) → XGBClassifier")
print(f"\n   Paramètres XGBClassifier:")
for param, value in xgb_params.items():
    print(f"      {param:20s}: {value}")

# =================================================================================
# ÉTAPE 5: ENTRAÎNEMENT
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 5: ENTRAÎNEMENT")
print(f"{'=' * 90}")

sw = compute_sample_weight('balanced', y_train)

print(f"\n   Pondération des classes : compute_sample_weight('balanced')")
print(f"   Pistes en entraînement  : {len(X_train_raw)}")
print(f"\n⏳ Entraînement en cours...")

start_train = time.time()
pipe_xgb.fit(X_train_raw, y_train, clf__sample_weight=sw)
duration_train = time.time() - start_train

print(f"✅ Entraînement terminé en {duration_train:.2f}s ({len(X_train_raw)/duration_train:.0f} pistes/s)")

# =================================================================================
# ÉTAPE 6: PRÉDICTIONS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 6: PRÉDICTIONS SUR LE SET DE TEST")
print(f"{'=' * 90}")

start_pred  = time.time()
preds       = pipe_xgb.predict(X_test_raw)
preds_proba = pipe_xgb.predict_proba(X_test_raw)
duration_pred = time.time() - start_pred

print(f"\n   Pistes prédites    : {len(preds)}")
print(f"   Temps              : {duration_pred:.3f}s")
print(f"   Confiance moyenne  : {preds_proba.max(axis=1).mean():.4f}")
print(f"   Confiance min/max  : {preds_proba.max(axis=1).min():.4f} / {preds_proba.max(axis=1).max():.4f}")

# =================================================================================
# ÉTAPE 7: ÉVALUATION
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 7: ÉVALUATION")
print(f"{'=' * 90}")

f1_test       = f1_score(y_test, preds, average="macro")
accuracy_test = (preds == y_test).sum() / len(y_test)
bal_acc       = balanced_accuracy_score(y_test, preds)

print(f"\n   ╔══════════════════════════════════════╗")
print(f"   ║  F1 Macro          : {f1_test:.4f}          ║")
print(f"   ║  Accuracy          : {accuracy_test:.4f}          ║")
print(f"   ║  Balanced Accuracy : {bal_acc:.4f}          ║")
print(f"   ╚══════════════════════════════════════╝")

# =================================================================================
# ÉTAPE 8: VALIDATION CROISÉE
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 8: VALIDATION CROISÉE — GroupKFold(5)")
print(f"{'=' * 90}")

gkf = GroupKFold(n_splits=CV_FOLDS)

print(f"\n⏳ Validation croisée en cours...")
start_cv  = time.time()
cv_scores = []

for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train_raw, y_train, groups=groups_train), 1):
    X_fold_train = X_train_raw.iloc[tr_idx]
    X_fold_val   = X_train_raw.iloc[val_idx]
    y_fold_train = y_train[tr_idx]
    y_fold_val   = y_train[val_idx]

    sw_fold   = compute_sample_weight('balanced', y_fold_train)
    pipe_fold = clone(pipe_xgb)
    pipe_fold.fit(X_fold_train, y_fold_train, clf__sample_weight=sw_fold)

    preds_fold = pipe_fold.predict(X_fold_val)
    score      = f1_score(y_fold_val, preds_fold, average='macro')
    cv_scores.append(score)
    print(f"   Fold {fold_idx}: {score:.4f}")

cv_scores   = np.array(cv_scores)
duration_cv = time.time() - start_cv

print(f"\n   Terminé en {duration_cv:.2f}s")
print(f"\n   Moyenne : {cv_scores.mean():.4f}")
print(f"   Std dev : {cv_scores.std():.4f}")
print(f"   Min     : {cv_scores.min():.4f}")
print(f"   Max     : {cv_scores.max():.4f}")

# =================================================================================
# ÉTAPE 9: RAPPORT PAR GENRE
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 9: RAPPORT PAR GENRE")
print(f"{'=' * 90}\n")

print(classification_report(y_test, preds, target_names=le.classes_, digits=4))

# =================================================================================
# ÉTAPE 10: ANALYSE DES CONFUSIONS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 10: ANALYSE DES CONFUSIONS")
print(f"{'=' * 90}")

cm         = confusion_matrix(y_test, preds)
errors     = (preds != y_test).sum()
error_rate = errors / len(y_test) * 100

print(f"\n   Prédictions correctes   : {len(y_test) - errors} / {len(y_test)} ({100-error_rate:.1f}%)")
print(f"   Prédictions incorrectes : {errors} / {len(y_test)} ({error_rate:.1f}%)")

print(f"\n   Paires les plus confondues:")
confusion_pairs = []
for i in range(len(le.classes_)):
    for j in range(i+1, len(le.classes_)):
        if cm[i, j] + cm[j, i] > 0:
            confusion_pairs.append((le.classes_[i], le.classes_[j], cm[i, j] + cm[j, i]))
confusion_pairs.sort(key=lambda x: x[2], reverse=True)
for gen1, gen2, count in confusion_pairs[:5]:
    print(f"      {gen1:15s} ↔ {gen2:15s}: {count:3d} confusions")

# =================================================================================
# ÉTAPE 11: IMPORTANCE DES FEATURES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 11: IMPORTANCE DES FEATURES")
print(f"{'=' * 90}")

xgb_clf    = pipe_xgb.named_steps['clf']
importances = xgb_clf.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature'   : X_train_raw.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(f"\n   Top 15 features:")
for _, row in feature_importance_df.head(15).iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"   {row['Feature']:30s} {bar} {row['Importance']:.4f}")

# =================================================================================
# ÉTAPE 12: MATRICE DE CONFUSION
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 12: MATRICE DE CONFUSION")
print(f"{'=' * 90}")

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format='d')
plt.title(f"Matrice de Confusion — XGBoost V1 (F1={f1_test:.4f})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =================================================================================
# ÉTAPE 13: TABLEAU DE RÉSULTATS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 13: TABLEAU DE RÉSULTATS")
print(f"{'=' * 90}\n")

results_comparison = pd.DataFrame([{
    'Modèle'                  : 'XGBoost V1',
    'F1 Macro (test)'         : f"{f1_test:.4f}",
    'Accuracy (test)'         : f"{accuracy_test:.4f}",
    'Balanced Accuracy (test)': f"{bal_acc:.4f}",
    'F1 CV (moyenne)'         : f"{cv_scores.mean():.4f}",
    'F1 CV (std dev)'         : f"{cv_scores.std():.4f}",
    'Temps entraînement (s)'  : f"{duration_train:.2f}",
    'Seed'                    : SEED,
    'Split'                   : f"{TRAIN_SIZE*100:.0f}/20",
    'CV Folds'                : f"GroupKFold({CV_FOLDS})",
    'Preprocessing'           : 'SimpleImputer(median)',
    'Scaling'                 : 'Aucun',
    'n_estimators'            : xgb_params['n_estimators'],
    'Pistes train'            : len(X_train_raw),
    'Pistes test'             : len(X_test_raw),
    'Features'                : X_train_raw.shape[1],
}])

print(results_comparison.to_string(index=False))

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
results_comparison.to_csv(RESULTS_PATH, index=False)
print(f"\n   Résultats sauvegardés : {RESULTS_PATH}")

# =================================================================================
# RÉSUMÉ FINAL
# =================================================================================
print(f"\n{'=' * 90}")
print("RÉSUMÉ FINAL")
print(f"{'=' * 90}\n")

print(f"   F1 Macro (test)         : {f1_test:.4f}")
print(f"   Accuracy (test)         : {accuracy_test:.4f}")
print(f"   Balanced Accuracy       : {bal_acc:.4f}")
print(f"   F1 CV moyenne           : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Temps entraînement      : {duration_train:.2f}s")
print(f"   Features utilisées      : {X_train_raw.shape[1]}")
print(f"   Pistes train / test     : {len(X_train_raw)} / {len(X_test_raw)}")

print(f"\n{'=' * 90}")
print("✅ PIPELINE XGBOOST V1 TERMINÉ")
print(f"{'=' * 90}\n")