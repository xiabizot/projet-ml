# Dream Stream — Classification de genres musicaux (FMA Small)

**Projet ML2 — DU SDA — Promotion 007 — Mars 2026**

---

## Contexte

En streaming musical, prédire uniquement un genre principal est limité car les genres se chevauchent et les labels sont parfois ambigus.
Jusqu’où peut-on classifier automatiquement à partir de features audio MP3, et comment exploiter le mismatch genre principal / sous-genres pour proposer un étiquetage plus informatif (top-k / multi-label) et interpréter les erreurs ?

Le but : Classifier automatiquement le genre musical (`genre_top`, 8 classes) de morceaux du dataset **FMA Small** (8 000 pistes, 30s, MP3).

Deux specificites orientent toute l'approche :
- **Biais artiste** — le split officiel FMA ne garantit pas l'isolation par artiste. Solution : `GroupShuffleSplit` par artiste (seed=42, test=20%)
- **Mismatch labels** — 41,8% des pistes ont un `genre_top` absent de leurs sous-genres (label noise documente par les auteurs du dataset)

---

## Architecture du projet en 2 couches

Le projet est structure en deux niveaux, materialises in fine par un agent IA double :

```
COUCHE 1 — Agent N.1 (classification audio + recommandation simplifiée)
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

## Perimetres du projet

### Projet evalue (niveau 1)

Les notebooks suivants constituent le perimetre réel du projet ML soumis a evaluation du DU SDA:

| Notebook | Contenu |
|----------|---------|
| NB1 | EDA, exploration du dataset FMA Small |
| NB2 | Extraction des 351 features audio (librosa) |
| NB2BIS | Comparaison features V1 FMA vs V2 maison |
| NB3 | Logistic Regression + Random Forest (GridSearchCV) |
| NB3BIS | XGBoost (GridSearchCV) |
| NB3TER | MLP sklearn (GridSearchCV) |
| NB4 | CNN log-mel (PyTorch) |
| NB5 | Analyse mismatch + classification multi-label |
| NB6 | Comparaison globale des modeles + recommandation cosinus |

L'application Streamlit en niveau 1 est proposee en demonstration comme prolongement applicatif du pipeline de classification et de recommandation par similarite acoustique.

Comme suit :
- XGBoost → classification genre_top (F1 = 0.491, features audio 351D)
- Embeddings CNN 256D → recommandation cosinus (5 pistes similaires, précision 50.3%)
- Flag mismatch → signale si le genre_top est absent des sous-genres

### Explorations complementaires (niveau 2 hors perimetre evalue)

Les notebooks suivants sont des explorations complémentaires, non soumises a evaluation :

| Notebook | Contenu |
|----------|---------|
| NB7 / NB7_8000 | Transfer Learning PANNs (prototype + full) |
| NB8 | NLP sur metadonnees textuelles (TF-IDF) |
| NB6BIS | Comparaison tous modeles + curation (IsolationForest) |
| NB6TER | Interpretabilite (SHAP, Grad-CAM, biais corpus) |
| NB9 | Agent IA double + export artefacts Streamlit |

L'application Streamlit ajoute en niveau 2 les modeles de transfer learning (PANNs), le NLP sur metadonnees et l'interpretabilite SHAP pour enrichir la classification. Chaque modèle vote pour un genre. Le vote de chacun compte proportionnellement à sa performance (score F1) : le meilleur modèle a plus de poids. Les votes sont normalisés pour que le total fasse 100%.

Comme suit :
- XGBoost → classification genre_top (poids 32% dans l'ensemble)
- PANNs → Agent V2 utilise XGBoost PANNs (poids 38% dans l'ensemble)
- NLP → Agent V2 utilise LogReg TF-IDF (poids 30% dans l'ensemble)
- SHAP → onglet Explicabilité, barres d'impact par feature
- Grad-CAM → onglet Explicabilité, spectrogrammes CNN
- Recommandation cosinus → onglet Recommandation, embeddings CNN 256D
- Claude API → explication en langage naturel

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

Conclusion principale :
Le facteur limitant n'est pas le modèle mais la représentation :
- features agrégées → plafond F1 ~0.49
- spectrogrammes CNN → 0.53
- embeddings pré-entraînés PANNs → 0.61

La performance d'un modèle dépend moins de l'algorithme que de la représentation des données : 4 modèles tabulaires plafonnent au même score malgré des architectures très différentes, tandis que le passage aux spectrogrammes (CNN) puis aux embeddings pré-entraînés (PANNs) débloque successivement ce plafond.

---

## Installation

```bash
git clone <repo> && cd PROJET
pip install -r requirements.txt

# Lancer l'application Streamlit :
streamlit run app_streamlit.py
```

L'app Streamlit fonctionne immediatement apres le clone (les modeles .joblib sont dans le repo).

Pour les fonctionnalites avancees :

```bash
# GPU (NB4 CNN, NB7 PANNs) :
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Cle API Anthropic (optionnel, pour analyse Claude dans Streamlit) :
# Creer un fichier .env a la racine :
# ANTHROPIC_API_KEY=sk-ant-...

# Ecoute audio et Grad-CAM :
# Necessitent les dossiers data/ et spectrogrammes/ (non versionnes, trop lourds)
# A recuperer separement (Google Drive ou copie locale)
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
NB6TER                             Interpretabilite SHAP + GradCAM (~10min, GPU pour GradCAM)
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
