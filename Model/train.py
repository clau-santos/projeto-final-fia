    # -*- coding: utf-8 -*-
"""
train.py — Treina o modelo de Probabilidade de Default (PD).
Projeto Final FIA/LABDATA — Sistema Inteligente de Limite de Credito (Dia 3).

Entrada : Dados/abt.csv (ABT pronta, 100% numerica)
Saidas  : Model/model_pd.pkl        (melhor modelo)
          Model/metrics.json        (metricas dos modelos)
          Model/test_predictions.csv(PD no teste, p/ avaliacao e politica)

Fluxo:
  1. Carrega a ABT e separa X (features), y (TARGET) e o id.
  2. Split treino/teste estratificado (mantem a proporcao de inadimplencia).
  3. Treina dois modelos:
       - Regressao Logistica (baseline interpretavel, com balanceamento de classe);
       - HistGradientBoosting (modelo principal, com early stopping p/ overfitting).
  4. Avalia no teste (AUC, KS, Gini) e escolhe o de maior AUC.
  5. Salva o melhor modelo, as metricas e as predicoes de teste.

Uso:  python train.py
Requer: scikit-learn, pandas, numpy, joblib, pyyaml  (ver requirements.txt)
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(Path(__file__).resolve().parent / "config.yaml", encoding="utf-8"))
ABT = ROOT / CFG["paths"]["abt"]
ID, TGT = CFG["columns"]["id"], CFG["columns"]["target"]
M = CFG["model"]


def ks_gini(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    ks = float(np.max(tpr - fpr))
    auc = roc_auc_score(y_true, proba)
    return auc, ks, float(2 * auc - 1)


def main():
    if not ABT.exists():
        sys.exit(f"[ERRO] ABT nao encontrada: {ABT}\nRode antes o 2_abt_transform.py")

    print(f"[1/5] Lendo ABT: {ABT.name}")
    df = pd.read_csv(ABT)
    y = df[TGT].astype(int)
    X = df.drop(columns=[TGT, ID])
    print(f"      -> {X.shape[0]:,} linhas x {X.shape[1]} features | inadimplencia {y.mean():.2%}")

    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(
        X, y, df[ID], test_size=M["test_size"], random_state=M["random_state"], stratify=y)
    print(f"[2/5] Split: treino {len(Xtr):,} | teste {len(Xte):,}")


    sw = compute_sample_weight("balanced", ytr)  # corrige desbalanceamento

    print("[3/5] Treinando Regressao Logistica (baseline)...")
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=M["lr_max_iter"],
                                              class_weight="balanced"))])
    lr.fit(Xtr, ytr)
    pd_lr = lr.predict_proba(Xte)[:, 1]
    auc_lr, ks_lr, gini_lr = ks_gini(yte, pd_lr)
    print(f"      LR  -> AUC {auc_lr:.4f} | KS {ks_lr:.4f}")

    print("[4/5] Treinando HistGradientBoosting (principal)...")
    hgb = HistGradientBoostingClassifier(
        max_iter=M["hgb_max_iter"], learning_rate=M["hgb_learning_rate"],
        max_depth=M["hgb_max_depth"], l2_regularization=M["hgb_l2"],
        early_stopping=True, validation_fraction=0.1,
        random_state=M["random_state"])
    hgb.fit(Xtr, ytr, sample_weight=sw)
    pd_hgb = hgb.predict_proba(Xte)[:, 1]
    auc_hgb, ks_hgb, gini_hgb = ks_gini(yte, pd_hgb)
    print(f"      HGB -> AUC {auc_hgb:.4f} | KS {ks_hgb:.4f}")

    # escolhe o melhor por AUC
    if auc_hgb >= auc_lr:
        best_name, best_model, pd_best = "HistGradientBoosting", hgb, pd_hgb
    else:
        best_name, best_model, pd_best = "LogisticRegression", lr, pd_lr

    # limiar otimo (Youden J = KS)
    fpr, tpr, thr = roc_curve(yte, pd_best)
    thr_opt = float(thr[np.argmax(tpr - fpr)])

    metrics = {
        "best_model": best_name,
        "threshold_otimo": round(thr_opt, 4),
        "logistic_regression": {"auc": round(auc_lr, 4), "ks": round(ks_lr, 4), "gini": round(gini_lr, 4)},
        "hist_gradient_boosting": {"auc": round(auc_hgb, 4), "ks": round(ks_hgb, 4), "gini": round(gini_hgb, 4)},
        "n_treino": int(len(Xtr)), "n_teste": int(len(Xte)),
        "taxa_inadimplencia": round(float(y.mean()), 4),
        "features": list(X.columns),
        "defaults": X.median(numeric_only=True).to_dict(),
    }

    # importancia por permutacao (top 15) numa subamostra do teste
    sub = min(3000, len(Xte))
    pi = permutation_importance(best_model, Xte.iloc[:sub], yte.iloc[:sub],
                                n_repeats=10, random_state=M["random_state"], scoring="roc_auc")
    imp = sorted(zip(X.columns, pi.importances_mean), key=lambda t: -t[1])[:15]
    metrics["top_features"] = [[f, round(float(v), 5)] for f, v in imp]

    print("[5/5] Salvando modelo, metricas e predicoes de teste...")
    (ROOT / CFG["paths"]["model"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ROOT / CFG["paths"]["model"])
    json.dump(metrics, open(ROOT / CFG["paths"]["metrics"], "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    out = pd.DataFrame({ID: idte.values, TGT: yte.values, "PD": pd_best})
    out[CFG["columns"]["income"]] = Xte[CFG["columns"]["income"]].values  # renda p/ politica
    out.to_csv(ROOT / CFG["paths"]["test_predictions"], index=False)

    print(f"\n[OK] Melhor modelo: {best_name} (AUC {max(auc_lr, auc_hgb):.4f})")
    print(f"     Salvo em: {CFG['paths']['model']}")


if __name__ == "__main__":
    main()
