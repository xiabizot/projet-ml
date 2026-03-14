# Projet ML2 — Classification de genres musicaux (FMA)

Projet de groupe — Master Data Science & AI, Machine Learning 2 (2025-2026).

## Objectif

Classifier automatiquement le genre musical (`genre_top`, 8 classes) de morceaux du dataset **FMA Small** (8 000 tracks, 30 s, MP3).

## Dataset

**FMA: A Dataset for Music Analysis** — Defferrard et al., ISMIR 2017
- 8 000 morceaux, 8 genres équilibrés (~1 000 chacun)
- 6 tracks corrompues exclues → **7 994 exploitables**
- [GitHub FMA](https://github.com/mdeff/fma) | [Paper (arXiv)](https://arxiv.org/abs/1612.01840)

## Problématique

Le dataset FMA présente deux spécificités qui orientent notre approche :

1. **Fuite par artiste** — Les splits officiels du paper (colonne `set.split`) ne garantissent pas l'isolation par artiste. Un même artiste peut figurer dans le train et le test, gonflant artificiellement les scores. Nous imposons un **`GroupShuffleSplit` par artiste** pour éliminer ce biais — un choix méthodologique délibéré, supérieur aux splits du paper.

2. **Ambiguïté des sous-genres (mismatch)** — Certains morceaux portent des sous-genres contradictoires avec leur `genre_top` (ex. un morceau Rock tagué "Ambient", "Drone"). Cette incohérence dans les labels dégrade la classification. Nous formalisons et quantifions cet effet : les morceaux *cohérents* sont systématiquement mieux classés que les morceaux *ambigus*, quel que soit le modèle.

Ces deux constats guident notre **protocole commun** et motivent l'exploration de plusieurs paradigmes complémentaires pour tester les limites de la classification sur ce dataset.

## Protocole commun

| Paramètre | Valeur |
|---|---|
| Split | `GroupShuffleSplit` par artiste (seed=42, test=20 %) |
| Validation | `GridSearchCV` + `GroupKFold(5)` |
| Métrique principale | **F1 macro** |
| Preprocessing | `SimpleImputer(median)` → `RobustScaler` |
| Reproductibilité | Seed fixe, indices train/test sauvegardés, CSV protocole 15 colonnes |

## Résultats

| Modèle | Notebook | F1 macro | Accuracy | Bal. Acc |
|--------|----------|----------|----------|----------|
| CNN log-mel | NB4 | **0.5324** | 0.5458 | 0.5289 |
| XGBoost mismatch | NB5 | 0.5100 | 0.5201 | 0.5140 |
| XGBoost (GridSearch) | NB3BIS | 0.4907 | 0.5010 | 0.4946 |
| Logistic Regression | NB3 | 0.4656 | 0.4819 | 0.4702 |
| MLP (GridSearch) | NB3TER | 0.4644 | 0.4753 | 0.4705 |
| Random Forest | NB3 | 0.4580 | 0.4693 | 0.4709 |

**Constats clés :**
- Le CNN surpasse le plafond tabulaire en exploitant l'information temporelle des spectrogrammes
- **Pop** est le genre le plus difficile (recall < 0.25 pour tous les modèles) — acoustiquement trop générique
- L'hypothèse mismatch est **confirmée** sur les 4 modèles tabulaires (Δ F1 cohérents − ambigus > 0)
- Les features spectrales (spectral centroid, contrast, bandwidth) dominent l'interprétabilité XGBoost et LR

## Notebooks

| # | Notebook | Contenu |
|---|----------|---------|
| 1 | `NOTEBOOK1_EDA.ipynb` | Exploration du dataset FMA Small |
| 2 | `NOTEBOOK2_FEATURES.ipynb` | Extraction de 351 features audio (librosa) |
| 2bis | `NOTEBOOK2BIS_V1_V2.ipynb` | Comparaison features V1 natif dataset vs V2 calculés, sélection |
| 3 | `NOTEBOOK3_ML_TABULAIRE_BASE.ipynb` | Logistic Regression + Random Forest (GridSearchCV) |
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
