# Projet ML2 — Classification de genres musicaux (FMA)

Projet de groupe — Master Data Science & AI, Machine Learning 2 (2025-2026).

## Objectif

Classifier automatiquement le genre musical (`genre_top`, 8 classes) de morceaux issus du dataset **FMA Small** (8 000 tracks, 30 s, MP3) en explorant plusieurs approches complémentaires.

## Dataset

**FMA: A Dataset for Music Analysis** — Defferrard et al., ISMIR 2017
- 8 000 morceaux, 8 genres équilibrés (~1 000 chacun)
- 6 tracks corrompues exclues → 7 994 exploitables
- [GitHub FMA](https://github.com/mdeff/fma) | [Paper (arXiv)](https://arxiv.org/abs/1612.01840)

## Protocole commun

| Paramètre | Valeur |
|---|---|
| Split | `GroupShuffleSplit` par artiste (seed=42, test=20 %) |
| Validation | `GridSearchCV` + `GroupKFold(5)` |
| Métrique | F1 macro |
| Preprocessing | `SimpleImputer(median)` → `RobustScaler` |

## Notebooks

| # | Notebook | Contenu |
|---|----------|---------|
| 1 | `NOTEBOOK1_EDA.ipynb` | Exploration du dataset FMA |
| 2 | `NOTEBOOK2_FEATURES.ipynb` | Extraction de 351 features audio (librosa) |
| 2bis | `NOTEBOOK2BIS_V1_V2.ipynb` | Comparaison features V1 vs V2, sélection |
| 3 | `NOTEBOOK3_ML_TABULAIRE_BASE.ipynb` | Logistic Regression + Random Forest |
| 3bis | `NOTEBOOK3BIS_ML_TABULAIRE_XGBOOST.ipynb` | XGBoost (GridSearchCV) |
| 3ter | `NOTEBOOK3TER_MLP_TABULAIRE.ipynb` | MLP sklearn (GridSearchCV) |
| 4 | `NOTEBOOK4_DL_CNN.ipynb` | CNN sur spectrogrammes mel (PyTorch) |
| 5 | `NOTEBOOK5_MISMATCH_SUBGENRES.ipynb` | Hypothèse mismatch sous-genres |
| 6 | `NOTEBOOK6_COMPARAISON_GLOBALE.ipynb` | Comparaison globale (4 tabulaires + CNN) |

## Structure

```
outputs/
├── features/          # features_V2.csv, corrélations
├── resultats/         # CSV protocole (results_nb3.csv, ...)
├── cnn/               # Modèle CNN, courbes, indices
├── comparaison/       # Figures comparaison globale
├── mismatch/          # Figures mismatch sous-genres
└── mlp_tabulaire/     # Figures MLP
```

## Installation

```bash
pip install -r requirements.txt
```

Pour le GPU (PyTorch CUDA, notebook 4) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
