# Détection de fraude financière — IEEE-CIS Fraud Detection

Projet de Master Data Science : comparaison de trois modèles de Machine Learning — **Régression Logistique**, **Random Forest** et **XGBoost** — pour détecter les transactions frauduleuses, sélection du meilleur modèle selon un critère centré sur le **Recall**, puis interprétation par **SHAP**.

Ce dépôt couvre la **Partie II : Data Science** du plan (chapitres 7 à 10). La Partie I (Data Analytics, notebooks 01 et 02) est le point de départ ; la Partie III (application Streamlit) est l'étape suivante.

## Données

Dataset Kaggle *IEEE-CIS Fraud Detection* : `train_transaction.csv` (590 540 lignes, 394 colonnes) et `train_identity.csv` (144 233 lignes, 41 colonnes), fusionnés sur `TransactionID`. Taux de fraude : 3,5 % (≈ 27,6 transactions légitimes pour 1 fraude).

Les fichiers bruts ne sont pas versionnés : à placer dans `data/raw/`.

## Arborescence

```
ML_Project/
├── data/
│   ├── raw/                      
│   └── processed/         
├── notebooks/
│   ├── 01_Comprehension_Preparation.ipynb   
│   ├── 02_EDA_Statistiques_ACP.ipynb       
│   ├── 03_Preparation_ML.ipynb              
│   ├── 04_Model_Evaluation.ipynb           
│   └── 05_Model_Interpretability.ipynb    
├── scripts/
│   ├── train_logistic.py        
│   ├── train_random_forest.py    
│   └── train_xgboost.py        
├── models/                      
├── outputs/
│   ├── *_metrics.json            
│   ├── predictions/             
│   ├── figures/                  
│   ├── best_model.json          
│   └── *.csv                   
├── reports/                    
├── requirements.txt
└── README.md
```

| Dossier | Rôle |
|---|---|
| `data/raw` | Données sources, jamais modifiées |
| `data/processed` | Données nettoyées/encodées et jeux Train / Validation / Test |
| `notebooks` | Analyses documentées, structurées comme un rapport (Markdown + Code) |
| `scripts` | Entraînements reproductibles exécutables en ligne de commande |
| `models` | Artefacts de modélisation (modèles sérialisés, liste des variables) |
| `outputs` | Résultats générés : métriques, probabilités, figures, tableaux |
| `reports` | Livrables rédigés (rapport, présentation) |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution (ordre à respecter)

```bash
# 0. Placer df_prepared.parquet (sortie du notebook 01) dans data/processed/
#    (le notebook 01 l'écrit dans son dossier courant : le déplacer dans data/processed/)

# 1. variables, split, déséquilibre -> train/val/test.parquet
jupyter nbconvert --to notebook --execute notebooks/03_Preparation_ML.ipynb --inplace
#    ou ouvrir le notebook dans JupyterLab et exécuter toutes les cellules

# 2. entraînement des trois modèles (depuis la racine du projet)
python scripts/train_logistic.py
python scripts/train_random_forest.py
python scripts/train_xgboost.py

# 3. évaluation, sélection du modèle -> outputs/best_model.json
jupyter nbconvert --to notebook --execute notebooks/04_Model_Evaluation.ipynb --inplace

# 4. interprétabilité SHAP du modèle retenu
jupyter nbconvert --to notebook --execute notebooks/05_Model_Interpretability.ipynb --inplace
```

Les notebooks doivent être exécutés **depuis le dossier `notebooks/`** (les chemins sont relatifs : `..`). Durée indicative sur le jeu complet : notebook 03 de quelques minutes (SMOTE), Random Forest de plusieurs minutes, XGBoost de 1 à 5 minutes, calcul SHAP de quelques minutes.

## Choix méthodologiques

- **Sélection des variables** : non supervisée (aucune fuite). Retrait de `TransactionID`, de `TransactionDT` brut, des doublons de variables et des variables corrélées à plus de 0,95.
- **Séparation** : 70 / 15 / 15 %, stratifiée sur `isFraud`. La validation sert à régler (arrêt anticipé XGBoost, seuil de décision, sélection du modèle) ; le test n'est utilisé que pour l'évaluation finale.
- **Déséquilibre** : SMOTE, Random Undersampling et Class Weight sont comparés (notebook 03) ; **Class Weight** est retenu (aucune perte ni création de données, cohérent avec la priorité au Recall).
- **Métriques** : Accuracy, Precision, Recall, F1, ROC-AUC (tableau du plan) complétées par la PR-AUC et le F2-score, plus adaptés à une classe rare.
- **Seuil de décision** : en plus du seuil 0,5, un seuil optimal est choisi sur la validation en maximisant le **F2-score** (le Recall compte quatre fois plus que la Precision).
- **Choix du modèle** : F2 au seuil optimal sur la validation (PR-AUC en départage), confirmé sur le test. Le Recall est donc privilégié sans dégénérer en « tout classer comme fraude ».

## Limites connues

- Séparation **aléatoire** et non chronologique : les performances en production seront vraisemblablement plus faibles. Une validation temporelle est recommandée.
- Hyperparamètres fixés de manière raisonnable, sans recherche systématique (`RandomizedSearchCV`, Optuna).
- Imputation par la médiane calculée avant la séparation Train/Test (effet faible, à signaler).
- Les codes catégoriels (`card1`, `addr1`, `P_emaildomain`) sont traités comme numériques : acceptable pour les arbres, limitant pour la régression logistique.
- Un seul découpage, sans intervalle de confiance : des écarts très faibles entre modèles ne sont pas concluants.

## Étape suivante (Partie III)

Application Streamlit . Point d'attention : le modèle utilise plus de 200 variables, alors que l'interface ne demande que montant, produit et appareil ; les autres variables devront être complétées par des valeurs par défaut (médianes d'entraînement stockées avec `models/features.json`) ou un modèle réduit sera à envisager.
