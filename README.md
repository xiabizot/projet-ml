# Dream Stream Sciences — Classification de genres musicaux (FMA)

**Projet ML2 — Sorbonne Data Analytics — Promotion 007 — Mars 2026**

---

## Contexte

Classifier automatiquement le genre musical (`genre_top`, 8 classes) de morceaux du dataset **FMA Small** (8 000 pistes, 30s, MP3).

Deux specificites orientent toute l'approche :
- **Biais artiste** — le split officiel FMA ne garantit pas l'isolation par artiste. Solution : `GroupShuffleSplit` par artiste (seed=42, test=20%)
- **Mismatch labels** — 41,8% des pistes ont un `genre_top` absent de leurs sous-genres (label noise documente par les auteurs du dataset)

---

## Architecture en 2 couches

Le projet est structure en deux niveaux, materialises par un agent IA double :

```
COUCHE 1 — Agent N.1 (classification audio)
NB1 ──> NB2 ──> NB3/3BIS/3TER ──> NB4 ──> NB5 ──> NB6
EDA     Features   ML tabulaire     CNN    Mismatch  Comparaison
        351D       LR,RF,XGB,MLP   logmel  sublabels + Reco cosine

COUCHE 2 — Agent N.2 (enrichissement multimodal)
NB7 ──> NB8 ──> NB6BIS ──> NB6TER ──> NB9
PANNs   NLP     Tous       SHAP       Agent IA
2048D   TFIDF   modeles    GradCAM    + Streamlit
                + curation  biais
```

**AlgoRythms Agent N.1** : XGBoost sur features audio (351D) + recommandation cosine CNN

**AlgoRythms Agent N.2** : ensemble audio + NLP + PANNs + SHAP + curation + Claude API

---

## Resultats cles

| Modele | Notebook | F1 macro | Approche |
|--------|----------|----------|----------|
| XGBoost PANNs | NB7 (8000) | **0.609** | Transfer Learning |
| MLP PANNs | NB7 (8000) | 0.541 | Transfer Learning |
| CNN log-mel | NB4 | 0.532 | Deep Learning |
| LogReg TF-IDF NLP | NB8 | 0.527 | NLP (texte seul) |
| XGBoost mismatch | NB5 | 0.510 | Features audio |
| XGBoost GridSearch | NB3BIS | 0.491 | Features audio |
| Logistic Regression | NB3 | 0.466 | Features audio |
| MLP sklearn | NB3TER | 0.464 | Features audio |
| Random Forest | NB3 | 0.458 | Features audio |

**Enseignements** :
- Le plafond tabulaire (~0.49) est une limite de representation, pas de modele
- Le NLP bat l'audio tabulaire (F1 0.527 vs 0.491) — resultat inedit sur FMA
- Le transfer learning (PANNs) domine toutes les approches
- Pop est inclassable acoustiquement (genre commercial, pas sonore)

---

## Installation

```bash
git clone <repo> && cd PROJET
pip install -r requirements.txt

# GPU (NB4 CNN) :
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Cle API Anthropic (optionnel, pour explication Claude dans Streamlit) :
# Creer un fichier .env a la racine :
# ANTHROPIC_API_KEY=sk-ant-...
```

## Ordre d'execution

```
# Couche 1
NB1 -> NB2 -> NB2BIS               Fondations (~1h)
NB3 -> NB3BIS -> NB3TER            ML tabulaire (~30min)
NB4                                 CNN (~3h, GPU recommande)
NB5 -> NB6                         Mismatch + Comparaison V1 (~30min)

# Couche 2
NB7 -> NB7_8000                    Transfer Learning PANNs (~1h GPU)
NB8                                NLP (~20min)
NB6BIS                             Comparaison tous modeles + curation (~5min)
NB6TER                             Interpretabilite SHAP + GradCAM (~10min)
NB9                                Agent IA -> genere outputs/agent/

# Application
streamlit run app_streamlit.py
```

## Structure

```
PROJET/
├── NOTEBOOK1_EDA.ipynb
├── NOTEBOOK2_FEATURES.ipynb
├── NOTEBOOK2BIS_V1_V2.ipynb
├── NOTEBOOK3_ML_TABULAIRE_BASE.ipynb
├── NOTEBOOK3BIS_ML_TABULAIRE_XGBOOST.ipynb
├── NOTEBOOK3TER_MLP_TABULAIRE.ipynb
├── NOTEBOOK4_DL_CNN.ipynb
├── NOTEBOOK5_MISMATCH_SUBGENRES.ipynb
├── NOTEBOOK6_COMPARAISON_GLOBALE.ipynb
├── NOTEBOOK6BIS_COMPARAISON_TOUS_MODELES.ipynb
├── NOTEBOOK6TER_INTERPRETABILITE.ipynb
├── NOTEBOOK7_TRANSFER_LEARNING.ipynb
├── NOTEBOOK7_TRANSFER_LEARNING_8000.ipynb
├── NOTEBOOK8_NLP.ipynb
├── NOTEBOOK9_AGENT_IA.ipynb
│
├── app_streamlit.py                 Interface Streamlit (AlgoRythms)
├── src/
│   └── agent.py                     Logique metier agent IA
│
├── images/                          Visuels app Streamlit
│
├── outputs/
│   ├── features/                    features_V2.csv (351 features)
│   ├── resultats/                   CSV protocole (results_nb*.csv)
│   ├── cnn/                         Modele CNN + embeddings 256D
│   ├── comparaison/                 Graphiques NB6/NB6BIS
│   ├── interpretabilite/            SHAP + GradCAM (NB6TER)
│   ├── curation/                    Outliers IsolationForest (NB6BIS)
│   ├── transfer_learning/           PANNs embeddings + resultats (NB7)
│   ├── nlp/                         TF-IDF + ablation (NB8)
│   ├── mlp_tabulaire/               GridSearch MLP (NB3TER)
│   └── agent/                       Artefacts Streamlit (.joblib)
│
├── data/                            Donnees brutes (non versionne)
├── spectrogrammes/                  Spectrogrammes log-mel (non versionne)
├── requirements.txt
├── .gitignore
└── .env                             Cle API Anthropic (non versionne)
```

## Dataset

FMA: A Dataset for Music Analysis — Defferrard et al., ISMIR 2017
- GitHub : https://github.com/mdeff/fma
- Paper : https://arxiv.org/abs/1612.01840
- 6 pistes corrompues documentees (wiki errata) — retrouvees et confirmees dans NB1

## Decisions methodologiques

| Decision | Valeur | Justification |
|----------|--------|---------------|
| Split | GroupShuffleSplit par artiste | Evite le leakage artiste (superieur au split officiel FMA) |
| Seed | 42 | Reproductibilite |
| Metrique | F1 macro | Equilibre entre genres desequilibres |
| Imputation | SimpleImputer(median) via Pipeline | Pas de leakage train/test |
| Scaling | RobustScaler | Robuste aux outliers |
| Validation | GridSearchCV + GroupKFold(5) | Coherent avec le split principal |

## Application Streamlit

**AlgoRythms** — Agent Double IA Multimodal Musical
Developpe par Dream Stream Sciences

- Classification par 2 agents (audio seul vs ensemble multimodal)
- Ecoute du morceau
- Recommandation par similarite cosinus (embeddings CNN 256D)
- Explicabilite SHAP + Grad-CAM
- Upload MP3 externe
- Explication Claude API (optionnel)

```bash
streamlit run app_streamlit.py
```
