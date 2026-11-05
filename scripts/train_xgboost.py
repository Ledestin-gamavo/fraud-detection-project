"""
Chapitre 8 - Modèle 3 : XGBoost (meilleures performances attendues).

Entrée  : data/processed/{train,val,test}.parquet   (produits par 03_Preparation_ML.ipynb)
Sorties : models/xgboost.joblib
          outputs/xgboost_metrics.json
          outputs/predictions/xgboost_proba.parquet

Exécution : python scripts/train_xgboost.py
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
NOM = "XGBoost"

# ---------------------------------------------------------------- 1. Chargement
train = pd.read_parquet(ROOT / "data/processed/train.parquet")
val = pd.read_parquet(ROOT / "data/processed/val.parquet")
test = pd.read_parquet(ROOT / "data/processed/test.parquet")

splits = {}
for nom, df in [("train", train), ("validation", val), ("test", test)]:
    splits[nom] = (df.drop(columns="isFraud"), df["isFraud"])
X_train, y_train = splits["train"]
print(f"Train {X_train.shape} | Validation {splits['validation'][0].shape} | Test {splits['test'][0].shape}")

# ---------------------------------------------------------------- 2. Entraînement
# scale_pos_weight = n_légitimes / n_fraudes : équivalent XGBoost de class_weight="balanced".
# Early stopping sur la VALIDATION (métrique PR-AUC) : le nombre d'arbres est choisi automatiquement.
ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=ratio, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=50,
    n_jobs=-1, random_state=SEED,
)
t0 = time.time()
model.fit(X_train, y_train, eval_set=[splits["validation"]], verbose=100)
duree = time.time() - t0
print(f"Entraînement terminé en {duree:.1f} s | meilleure itération : {model.best_iteration}")

# ---------------------------------------------------------------- 3. Probabilités
probas = {nom: model.predict_proba(X)[:, 1] for nom, (X, y) in splits.items()}

# Seuil optimal : maximise le F2-score (le Recall pèse 4 fois plus que la Precision) sur la VALIDATION
y_val = splits["validation"][1]
prec, rec, seuils = precision_recall_curve(y_val, probas["validation"])
f2 = 5 * prec * rec / (4 * prec + rec + 1e-12)
seuil_opt = float(seuils[f2[:-1].argmax()])

# ---------------------------------------------------------------- 4. Métriques
metriques = {"modele": NOM, "duree_entrainement_s": round(duree, 1), "seuil_optimal_f2": seuil_opt,
             "meilleure_iteration": int(model.best_iteration)}
for nom, (X, y) in splits.items():
    metriques[nom] = {"roc_auc": roc_auc_score(y, probas[nom]),
                      "pr_auc": average_precision_score(y, probas[nom])}
    for etiquette, seuil in [("seuil_0.5", 0.5), ("seuil_optimal", seuil_opt)]:
        pred = (probas[nom] >= seuil).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        metriques[nom][etiquette] = {
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "f2": fbeta_score(y, pred, beta=2),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

# ---------------------------------------------------------------- 5. Sauvegardes
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "outputs/predictions").mkdir(parents=True, exist_ok=True)
joblib.dump(model, ROOT / "models/xgboost.joblib")
with open(ROOT / "outputs/xgboost_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=False)
pd.concat([pd.DataFrame({"split": nom, "y_true": y.values, "y_proba": probas[nom]})
           for nom, (X, y) in splits.items()]).to_parquet(ROOT / "outputs/predictions/xgboost_proba.parquet", index=False)

t = metriques["test"]
print(f"[TEST] ROC-AUC = {t['roc_auc']:.4f} | PR-AUC = {t['pr_auc']:.4f} | "
      f"Recall@0.5 = {t['seuil_0.5']['recall']:.4f} | Recall@seuil optimal ({seuil_opt:.3f}) = {t['seuil_optimal']['recall']:.4f}")
print("Modèle, métriques et probabilités sauvegardés.")
