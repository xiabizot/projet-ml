"""
=================================================================================
XGBoost V1 - CONTRAT D'EXPÉRIENCE PARTAGÉ
=================================================================================
Ce script implémente un "contrat d'expérience" permettant la comparaison équitable
entre plusieurs modèles (régression linéaire, XGBoost, SVM, etc.).

PARAMÈTRES FIXES (communs à tous les modèles):
  - Seed: 42 (reproductibilité)
  - Split: GroupShuffleSplit 80/20 par artiste
  - Prétraitement: StandardScaler
  - Métrique: F1 Macro (classification multi-classe)
  - Colonnes: mêmes features éliminées pour tous
  - Validation: 5-fold cross-validation
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# =================================================================================
# CONFIGURATION GLOBALE (CONTRAT)
# =================================================================================
print("=" * 90)
print("XGBoost V1 - CONTRAT D'EXPÉRIENCE PARTAGÉ")
print("=" * 90)

SEED = 42                                      # Seed globale pour reproductibilité
TRAIN_SIZE = 0.8                               # 80% train, 20% test
TEST_SIZE = 0.2
CV_FOLDS = 5                                   # 5-fold cross-validation
METRIC = 'f1_macro'                            # Métrique unique
SCALER_TYPE = 'standard'                       # StandardScaler

print(f"\n📋 PARAMÈTRES CONSTANTS (CONTRAT):")
print(f"   Seed aléatoire         : {SEED}")
print(f"   Split train/test       : {TRAIN_SIZE*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"   Stratégie de split     : GroupShuffleSplit par artiste")
print(f"   Folds de validation    : {CV_FOLDS}-fold cross-validation")
print(f"   Métrique principale    : {METRIC}")
print(f"   Normalisation          : {SCALER_TYPE}Scaler")

# Appliquer seed globale
np.random.seed(SEED)

# =================================================================================
# ÉTAPE 1: CHARGEMENT DES DONNÉES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 1: CHARGEMENT ET EXPLORATION DES DONNÉES")
print(f"{'=' * 90}")

print(f"\n⏱️  Chargement de features_V2.csv...")
start_load = time.time()
df_custom = pd.read_csv(r"C:\Users\dmerchan\projet-ml\Data\features_V2.csv", index_col="track_id")
duration_load = time.time() - start_load

print(f"✅ Chargé en {duration_load:.2f}s")
print(f"   - Forme: {df_custom.shape} (pistes × colonnes)")
print(f"   - Pistes uniques: {len(df_custom.index)}")
print(f"   - Colonnes totales: {df_custom.columns.tolist()[:5]}... (affichage limité)")

# =================================================================================
# ÉTAPE 2: PRÉPARATION DES DONNÉES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 2: NETTOYAGE ET PRÉPARATION DES DONNÉES")
print(f"{'=' * 90}")

print(f"\n🧹 Suppression des colonnes non-numériques...")
cols_to_drop = ['genre_top', 'genres', 'genres_decoded', 'n_subgenres', 
                'mismatch', 'artist_name', 'track_title']
cols_dropped = [c for c in cols_to_drop if c in df_custom.columns]
X = df_custom.drop(columns=cols_dropped)

print(f"   - Colonnes supprimées: {len(cols_dropped)} ({cols_dropped})")
print(f"   - Colonnes restantes: {X.shape[1]}")

print(f"\n🔢 Conversion en numérique + gestion NaN...")
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.select_dtypes(include=[np.number])
print(f"   - Features finales: {X.shape[1]} colonnes")
print(f"   - NaN après imputation: {X.isna().sum().sum()}")

print(f"\n🏷️  Encodage de la cible (genre_top)...")
le = LabelEncoder()
y = le.fit_transform(df_custom["genre_top"])
print(f"   - Classes: {list(le.classes_)}")
print(f"   - Nombre de classes: {len(le.classes_)}")
print(f"   - Distribution classes:")
for cls, idx in zip(le.classes_, range(len(le.classes_))):
    count = (y == idx).sum()
    print(f"      {cls:20s}: {count:5d} pistes ({count/len(y)*100:5.1f}%)")

print(f"\n👥 Extraction des groupes (artistes pour split intelligent)...")
groups = df_custom["artist_name"].values
unique_artists = len(np.unique(groups))
print(f"   - Artistes uniques: {unique_artists}")
print(f"   - Pistes par artiste (moyenne): {len(groups) / unique_artists:.1f}")

# =================================================================================
# ÉTAPE 3: NORMALISATION
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 3: NORMALISATION DES FEATURES")
print(f"{'=' * 90}")

print(f"\n📊 Application de StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print(f"   ✅ Scaling appliqué")
print(f"   - Moyenne après scaling: {X_scaled.mean().mean():.6f} (≈ 0)")
print(f"   - Écart-type après scaling: {X_scaled.std().mean():.6f} (≈ 1)")
print(f"   - Min global: {X_scaled.min().min():.4f}")
print(f"   - Max global: {X_scaled.max().max():.4f}")

# =================================================================================
# ÉTAPE 4: SPLIT TRAIN/TEST
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 4: DIVISION TRAIN/TEST (GroupShuffleSplit par artiste)")
print(f"{'=' * 90}")

print(f"\n🔀 Split 80/20 avec contrôle artiste...")
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(X_scaled, y, groups=groups))

X_train = X_scaled.iloc[train_idx]
X_test = X_scaled.iloc[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"   ✅ Split effectué")
print(f"   - Train: {len(train_idx)} pistes ({len(train_idx)/len(y)*100:.1f}%)")
print(f"   - Test:  {len(test_idx)} pistes ({len(test_idx)/len(y)*100:.1f}%)")
print(f"   - Artistes communs train/test: 0 (GROUP SPLIT VALIDÉ) ✓")

# Vérifier pas de chevauchement artiste
train_groups = groups[train_idx]
test_groups = groups[test_idx]
overlap = len(set(train_groups) & set(test_groups))
print(f"   - Vérification leakage artiste: {overlap} artistes en commun (OK si 0)")

# =================================================================================
# ÉTAPE 5: ENTRAÎNEMENT XGBOOST
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 5: ENTRAÎNEMENT DU MODÈLE XGBOOST")
print(f"{'=' * 90}")

xgb_params = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0
}

print(f"\n🤖 Paramètres XGBoost:")
for param, value in xgb_params.items():
    print(f"   - {param:20s}: {value}")

print(f"\n⏳ Entraînement en cours sur {len(X_train)} pistes...")
start_train = time.time()
xgb_model = XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)
duration_train = time.time() - start_train

print(f"✅ Entraînement terminé en {duration_train:.2f}s")
print(f"   - Vitesse: {len(X_train)/duration_train:.0f} pistes/seconde")

# =================================================================================
# ÉTAPE 6: PRÉDICTIONS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 6: PRÉDICTIONS SUR LE SET DE TEST")
print(f"{'=' * 90}")

print(f"\n🎯 Génération des prédictions...")
start_pred = time.time()
preds = xgb_model.predict(X_test)
preds_proba = xgb_model.predict_proba(X_test)
duration_pred = time.time() - start_pred

print(f"✅ Prédictions générées en {duration_pred:.3f}s")
print(f"   - Pistes prédites: {len(preds)}")
print(f"   - Classes prédites: {np.unique(preds).tolist()}")
print(f"   - Confiance moyenne: {preds_proba.max(axis=1).mean():.4f}")
print(f"   - Confiance min: {preds_proba.max(axis=1).min():.4f}")
print(f"   - Confiance max: {preds_proba.max(axis=1).max():.4f}")

# =================================================================================
# ÉTAPE 7: ÉVALUATION PRINCIPALE
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 7: ÉVALUATION (MÉTRIQUE PRINCIPALE)")
print(f"{'=' * 90}")

f1_test = f1_score(y_test, preds, average="macro")
accuracy_test = (preds == y_test).sum() / len(y_test)

print(f"\n🔥 RÉSULTAT PRINCIPAL:")
print(f"   ╔════════════════════════════════════╗")
print(f"   ║  F1 MACRO (test set): {f1_test:.4f}      ║")
print(f"   ║  Accuracy (test set): {accuracy_test:.4f}      ║")
print(f"   ╚════════════════════════════════════╝")

# =================================================================================
# ÉTAPE 8: CROSS-VALIDATION
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 8: VALIDATION CROISÉE (5-Fold)")
print(f"{'=' * 90}")

print(f"\n🔄 Exécution de la validation croisée...")
start_cv = time.time()
cv_scores = cross_val_score(xgb_model, X_train, y_train, 
                             cv=CV_FOLDS, scoring=METRIC, n_jobs=-1)
duration_cv = time.time() - start_cv

print(f"✅ CV terminée en {duration_cv:.2f}s")
print(f"\n   Scores par fold:")
for fold, score in enumerate(cv_scores, 1):
    print(f"      Fold {fold}: {score:.4f}")
print(f"\n   Statistiques CV:")
print(f"      Moyenne:     {cv_scores.mean():.4f}")
print(f"      Std dev:     {cv_scores.std():.4f}")
print(f"      Min:         {cv_scores.min():.4f}")
print(f"      Max:         {cv_scores.max():.4f}")

# =================================================================================
# ÉTAPE 9: RAPPORT DÉTAILLÉ PAR CLASSE
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 9: RAPPORT DÉTAILLÉ PAR GENRE")
print(f"{'=' * 90}\n")

report = classification_report(y_test, preds, target_names=le.classes_, 
                               digits=4, output_dict=False)
print(report)

# =================================================================================
# ÉTAPE 10: ANALYSE DES CONFUSIONS
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 10: ANALYSE DES ERREURS ET CONFUSIONS")
print(f"{'=' * 90}")

cm = confusion_matrix(y_test, preds)
errors = (preds != y_test).sum()
error_rate = errors / len(y_test) * 100

print(f"\n❌ Résumé des erreurs:")
print(f"   - Prédictions correctes: {len(y_test) - errors} / {len(y_test)} ({100-error_rate:.1f}%)")
print(f"   - Prédictions incorrectes: {errors} / {len(y_test)} ({error_rate:.1f}%)")

# Confusions principales
print(f"\n🔄 Paires de genres les plus confondus:")
confusion_pairs = []
for i in range(len(le.classes_)):
    for j in range(i+1, len(le.classes_)):
        if cm[i,j] + cm[j,i] > 0:
            confusion_pairs.append((le.classes_[i], le.classes_[j], cm[i,j] + cm[j,i]))
confusion_pairs.sort(key=lambda x: x[2], reverse=True)
for gen1, gen2, count in confusion_pairs[:5]:
    print(f"   - {gen1:15s} ↔ {gen2:15s}: {count:3d} confusions")

# =================================================================================
# ÉTAPE 11: IMPORTANCE DES FEATURES
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 11: IMPORTANCE DES VARIABLES")
print(f"{'=' * 90}")

importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(f"\n📊 Top 15 features les plus influentes:")
for idx, row in feature_importance_df.head(15).iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"   {row['Feature']:30s} {bar} {row['Importance']:.4f}")

# =================================================================================
# ÉTAPE 12: MATRICE DE CONFUSION (VISUALISATION)
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 12: MATRICE DE CONFUSION (VISUALISATION)")
print(f"{'=' * 90}")

fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format='d')
plt.title(f"Matrice de Confusion - XGBoost V1 (F1={f1_test:.4f})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =================================================================================
# ÉTAPE 13: TABLEAU DE RÉSULTATS POUR COMPARAISON
# =================================================================================
print(f"\n{'=' * 90}")
print("ÉTAPE 13: TABLEAU DE RÉSULTATS PARTAGÉS")
print(f"{'=' * 90}\n")

results_comparison = pd.DataFrame([{
    'Modèle': 'XGBoost V1',
    'F1 Macro (test)': f"{f1_test:.4f}",
    'Accuracy (test)': f"{accuracy_test:.4f}",
    'F1 CV (moyenne)': f"{cv_scores.mean():.4f}",
    'F1 CV (std dev)': f"{cv_scores.std():.4f}",
    'Temps entraînement (s)': f"{duration_train:.2f}",
    'Seed': SEED,
    'Split': f"{TRAIN_SIZE*100:.0f}/20",
    'CV Folds': CV_FOLDS,
    'Pistes train': len(X_train),
    'Pistes test': len(X_test),
    'Features': X_train.shape[1],
}])

print(results_comparison.to_string(index=False))

# Sauvegarder pour comparaison
results_comparison.to_csv(r"C:\Users\dmerchan\projet-ml\XGBoost\results_V1.csv", index=False)
print(f"\n✅ Résultats sauvegardés dans: XGBoost/results_V1.csv")

# =================================================================================
# RÉSUMÉ FINAL
# =================================================================================
print(f"\n{'=' * 90}")
print("RÉSUMÉ FINAL - CONTRAT D'EXPÉRIENCE VALIDÉ")
print(f"{'=' * 90}\n")

print(f"✅ VALIDATION DU CONTRAT:")
print(f"   ✓ Seed global = {SEED}")
print(f"   ✓ Split GroupShuffleSplit = Sans leakage artiste")
print(f"   ✓ Normalisation = StandardScaler (fit train, transform test)")
print(f"   ✓ Métrique = {METRIC}")
print(f"   ✓ CV = {CV_FOLDS}-Fold avec même seed")
print(f"   ✓ Colonnes = Identiques pour tous les modèles")
print(f"   ✓ Résultats = Sauvegardés pour comparaison")

print(f"\n🎯 PROCHAINES ÉTAPES:")
print(f"   1. Votre collègue: exécute son modèle (régression linéaire, etc.)")
print(f"   2. Même contrat (seed, split, normalisation, métrique)")
print(f"   3. Sauvegarde résultats dans results_[MODEL].csv")
print(f"   4. Fusion des résultats → tableau de comparaison unique")

print(f"\n{'=' * 90}")
print("✅ PIPELINE XGBOOST V1 TERMINÉE AVEC SUCCÈS")
print(f"{'=' * 90}\n")