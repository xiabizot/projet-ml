"""
=================================================================================
XGBoost V2 - CONTRAT D'EXPÉRIENCE PARTAGÉ (GRIDSEARCH)
=================================================================================
PARAMÈTRES FIXES (communs à tous les modèles):
  - Seed              : 42
  - Split             : GroupShuffleSplit 80/20 par artiste (indices sauvegardés)
  - Preprocessing     : SimpleImputer(median) + RobustScaler dans Pipeline
  - Métrique          : F1 Macro
  - Features          : 351 features audio
  - Validation        : GroupKFold 5-fold (intégré dans GridSearchCV)
  - Pondération       : compute_sample_weight('balanced')
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, RobustScaler
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
RESULTS_PATH    = BASE / 'XGBoost' / 'results_V2.csv'

print("=" * 90)
print("XGBoost V2 - CONTRAT D'EXPÉRIENCE PARTAGÉ (GRIDSEARCH)")
print("=" * 90)
print(f"\n📋 PARAMÈTRES:")
print(f"   Seed                   : {SEED}")
print(f"   Split train/test       : {TRAIN_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Stratégie de split     : GroupShuffleSplit par artiste")
print(f"   Validation croisée     : GroupKFold ({CV_FOLDS} folds)")
print(f"   Métrique principale    : {METRIC}")
print(f"   Preprocessing          : SimpleImputer(median) + RobustScaler")
print(f"   Recherche paramètres   : GridSearchCV")

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
# ÉTAPE 3: DIVISION TRAIN/TEST ET SAUVEGARDE DES INDICES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 3: DIVISION TRAIN/TEST")
print(f"{'=' * 90}")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))

# Sauvegarde des indices avec vérification d'écrasement
data_dir = BASE / 'Data'
data_dir.mkdir(parents=True, exist_ok=True)

train_path = data_dir / 'train_idx.npy'
test_path = data_dir / 'test_idx.npy'

print("\n   Vérification des fichiers d'indices :")
if train_path.exists():
    print(f"   ⚠️ Écrasement du fichier existant : {train_path.name}")
if test_path.exists():
    print(f"   ⚠️ Écrasement du fichier existant : {test_path.name}")

np.save(train_path, train_idx)
np.save(test_path, test_idx)

X_train_raw  = X.iloc[train_idx]
X_test_raw   = X.iloc[test_idx]
y_train      = y_enc[train_idx]
y_test       = y_enc[test_idx]
groups_train = groups[train_idx]

overlap = len(set(groups[train_idx]) & set(groups[test_idx]))

print(f"\n   Train          : {len(train_idx)} pistes ({len(train_idx)/len(y_enc)*100:.1f}%)")
print(f"   Test           : {len(test_idx)} pistes ({len(test_idx)/len(y_enc)*100:.1f}%)")
print(f"   Artistes communs train/test : {overlap}")
print(f"   Indices sauvegardés dans : {data_dir}")

# =================================================================================
# ÉTAPE 4: PIPELINE DE BASE
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 4: CONSTRUCTION DU PIPELINE")
print(f"{'=' * 90}")

xgb_base_params = {
    "objective"     : "multi:softprob",
    "num_class"     : len(le.classes_),
    "tree_method"   : "hist",
    "random_state"  : SEED,
    "n_jobs"        : -1,
    "verbosity"     : 0
}

pipe_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()), # Ajouté pour respecter le contrat global
    ('clf',     XGBClassifier(**xgb_base_params))
])

print(f"\n   Pipeline       : SimpleImputer(median) → RobustScaler → XGBClassifier")

# =================================================================================
# ÉTAPE 5: ENTRAÎNEMENT & GRIDSEARCH
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 5: RECHERCHE D'HYPERPARAMÈTRES (GRIDSEARCH)")
print(f"{'=' * 90}")

# Grille de paramètres définie par l'équipe
param_grid = {
    'clf__n_estimators': [100, 300, 500],
    'clf__max_depth': [10, 20, 30],
    'clf__min_child_weight': [4, 8] # Équivalent XGBoost du 'min_samples_leaf'
}

gkf_grid = GroupKFold(n_splits=CV_FOLDS)

grid_search = GridSearchCV(
    estimator=pipe_xgb,
    param_grid=param_grid,
    cv=gkf_grid,
    scoring=METRIC,
    n_jobs=-1,
    verbose=2 # Permet de voir l'avancement dans la console
)

sw = compute_sample_weight('balanced', y_train)

print(f"\n   Pondération des classes : compute_sample_weight('balanced')")
print(f"   Pistes en entraînement  : {len(X_train_raw)}")
print(f"   Grille de paramètres    : {param_grid}")
print(f"\n⏳ Entraînement GridSearch en cours (cela peut prendre du temps)...")

start_train = time.time()
# On passe bien les groupes pour le GroupKFold interne, et les poids pour XGBoost
grid_search.fit(X_train_raw, y_train, groups=groups_train, clf__sample_weight=sw)
duration_train = time.time() - start_train

best_pipe_xgb = grid_search.best_estimator_

print(f"\n✅ Entraînement terminé en {duration_train:.2f}s")
print(f"   Meilleurs paramètres : {grid_search.best_params_}")
print(f"   Meilleur score CV F1 : {grid_search.best_score_:.4f}")

# =================================================================================
# ÉTAPE 6: PRÉDICTIONS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 6: PRÉDICTIONS SUR LE SET DE TEST")
print(f"{'=' * 90}")

start_pred  = time.time()
preds       = best_pipe_xgb.predict(X_test_raw)
preds_proba = best_pipe_xgb.predict_proba(X_test_raw)
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
# ÉTAPE 8: VALIDATION CROISÉE (DÉDUITE DU GRIDSEARCH)
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 8: VALIDATION CROISÉE — Résultats du meilleur modèle")
print(f"{'=' * 90}")

# Au lieu de relancer la CV complète, on extrait les scores du meilleur modèle testé par GridSearchCV
best_index = grid_search.best_index_
cv_results = grid_search.cv_results_
cv_scores = []

print(f"\n   Scores obtenus par le meilleur modèle lors des {CV_FOLDS} folds du GridSearch:")
for i in range(CV_FOLDS):
    fold_score = cv_results[f'split{i}_test_score'][best_index]
    cv_scores.append(fold_score)
    print(f"   Fold {i+1}: {fold_score:.4f}")

cv_scores = np.array(cv_scores)

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

# Extraction de XGBoost depuis le MEILLEUR pipeline
xgb_clf = best_pipe_xgb.named_steps['clf']
importances = xgb_clf.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature'   : X_train_raw.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(f"\n   Top 15 features (sur {len(X_train_raw.columns)}):")
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
plt.title(f"Matrice de Confusion — XGBoost V2 (F1={f1_test:.4f})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =================================================================================
# ÉTAPE 13: TABLEAU DE RÉSULTATS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 13: TABLEAU DE RÉSULTATS")
print(f"{'=' * 90}\n")

results_comparison = pd.DataFrame([{
    'Modèle'                  : 'XGBoost V2 (GridSearch)',
    'F1 Macro (test)'         : f"{f1_test:.4f}",
    'Accuracy (test)'         : f"{accuracy_test:.4f}",
    'Balanced Accuracy (test)': f"{bal_acc:.4f}",
    'F1 CV (moyenne)'         : f"{cv_scores.mean():.4f}",
    'F1 CV (std dev)'         : f"{cv_scores.std():.4f}",
    'Temps entraînement (s)'  : f"{duration_train:.2f}",
    'Seed'                    : SEED,
    'Split'                   : f"{TRAIN_SIZE*100:.0f}/20",
    'CV Folds'                : f"GroupKFold({CV_FOLDS})",
    'Preprocessing'           : 'SimpleImputer(median) + RobustScaler',
    'Best Parameters'         : str(grid_search.best_params_),
    'Pistes train'            : len(X_train_raw),
    'Pistes test'             : len(X_test_raw),
    'Features'                : X_train_raw.shape[1],
}])

print(results_comparison.to_string(index=False))

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

print("\n   Vérification du fichier de résultats :")
if RESULTS_PATH.exists():
    print(f"   ⚠️ Écrasement du fichier existant : {RESULTS_PATH.name}")

results_comparison.to_csv(RESULTS_PATH, index=False)
print(f"   ✅ Résultats sauvegardés : {RESULTS_PATH}")

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
print(f"   Temps GridSearchCV      : {duration_train:.2f}s")
print(f"   Meilleurs paramètres    : {grid_search.best_params_}")
print(f"   Features utilisées      : {X_train_raw.shape[1]}")
print(f"   Pistes train / test     : {len(X_train_raw)} / {len(X_test_raw)}")

print(f"\n{'=' * 90}")
print("✅ PIPELINE XGBOOST V2 TERMINÉ")
print(f"{'=' * 90}\n")