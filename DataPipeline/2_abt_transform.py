# -*- coding: utf-8 -*-
"""
abt_transform.py  —  Transformacao dos dados limpos na ABT (Analytical Base Table).
Projeto Final FIA/LABDATA · Home Credit Default Risk · Ponto iii

Entrada : Dados/clean_data.csv (saida do data_sanitization.py)
Saida   : Dados/abt.csv (tabela analitica, 100% numerica, sem nulos, pronta p/ modelagem)

Etapas (decisoes documentadas na EDA do Ponto ii):
  1. Descartar colunas com nulos acima do limite (config: null_drop_threshold).
  2. Engenharia de atributos: idade/tempo de emprego em anos, razoes financeiras
     (credito/renda, anuidade/renda, prazo), agregados de EXT_SOURCE.
  3. Encoding de categoricas:
        - alta cardinalidade  -> frequency encoding (config: freq_encode_cols)
        - demais ate o limite  -> one-hot encoding
  4. Imputacao: numericas pela mediana; indicadores ja tratados.
  5. Exportar abt.csv com ID + TARGET + features.

Uso:
  python abt_transform.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

CLEAN = ROOT / CFG["paths"]["clean"]
ABT   = ROOT / CFG["paths"]["abt"]
ID    = CFG["columns"]["id"]
TGT   = CFG["columns"]["target"]
A     = CFG["abt"]


def engenharia_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas com forte interpretacao de negocio."""
    eps = 1e-5  # evita divisao por zero
    # Tempo em anos (DAYS_* sao negativos: dias antes da solicitacao)
    df["AGE_YEARS"]        = (-df["DAYS_BIRTH"] / 365.25)
    df["YEARS_EMPLOYED"]   = (-df["DAYS_EMPLOYED"] / 365.25)
    df["EMPLOYED_TO_AGE"]  = df["YEARS_EMPLOYED"] / (df["AGE_YEARS"] + eps)
    # Razoes financeiras
    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]  / (df["AMT_INCOME_TOTAL"] + eps)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["CREDIT_TERM"]          = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)
    df["CREDIT_GOODS_RATIO"]   = df["AMT_CREDIT"]  / (df["AMT_GOODS_PRICE"] + eps)
    df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + eps)
    # Agregados dos scores externos
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
    df["EXT_SOURCE_STD"]  = df[ext].std(axis=1)
    df["EXT_SOURCE_MIN"]  = df[ext].min(axis=1)
    df["EXT_SOURCE_MAX"]  = df[ext].max(axis=1)
    return df


def main():
    if not CLEAN.exists():
        sys.exit(f"[ERRO] clean_data nao encontrado: {CLEAN}\n"
                 f"Rode antes: python data_sanitization.py")

    print(f"[1/6] Lendo base limpa: {CLEAN.name}")
    df = pd.read_csv(CLEAN)
    print(f"      -> {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    # guardar id e alvo, trabalhar nas features
    y  = df[TGT]
    ids = df[ID]
    X  = df.drop(columns=[TGT, ID])

    # 1. Descartar colunas muito vazias
    thr = A["null_drop_threshold"]
    null_ratio = X.isna().mean()
    drop_cols = null_ratio[null_ratio > thr].index.tolist()
    X = X.drop(columns=drop_cols)
    print(f"[2/6] Colunas descartadas por >{thr:.0%} nulos: {len(drop_cols)}")

    # 2. Engenharia de atributos
    X = engenharia_atributos(X)
    print(f"[3/6] Features derivadas criadas (idade, razoes, EXT_SOURCE_*)")

    # 3. Encoding de categoricas
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    freq_cols = [c for c in A["freq_encode_cols"] if c in cat_cols]
    # 3a. frequency encoding (alta cardinalidade)
    for c in freq_cols:
        freq = X[c].value_counts(normalize=True)
        X[c + "_FREQ"] = X[c].map(freq).fillna(0.0)
    X = X.drop(columns=freq_cols)
    # 3b. one-hot nas demais (dentro do limite de cardinalidade)
    onehot_cols = [c for c in cat_cols if c not in freq_cols
                   and X[c].nunique(dropna=True) <= A["onehot_max_cardinality"]]
    X = pd.get_dummies(X, columns=onehot_cols, dummy_na=True, dtype="int8")
    # qualquer categorica restante (acima do limite e nao listada) -> frequency
    resto = X.select_dtypes(include="object").columns.tolist()
    for c in resto:
        freq = X[c].value_counts(normalize=True)
        X[c + "_FREQ"] = X[c].map(freq).fillna(0.0)
    if resto:
        X = X.drop(columns=resto)
    print(f"[4/6] Encoding: {len(freq_cols)+len(resto)} frequency + {len(onehot_cols)} one-hot")

    # 4. Imputacao numerica pela mediana
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    # substituir eventuais infinitos por 0 (razoes com denominador minimo)
    X = X.replace([np.inf, -np.inf], 0.0)
    print(f"[5/6] Imputacao mediana aplicada; nulos restantes: {int(X.isna().sum().sum())}")

    # cast para float32 (opcional, reduz tamanho)
    if A.get("cast_float32", False):
        fcols = X.select_dtypes(include=["float64"]).columns
        X[fcols] = X[fcols].astype("float32")

    # 5. Montar ABT final = ID + TARGET + features
    abt = pd.concat([ids.reset_index(drop=True), y.reset_index(drop=True),
                     X.reset_index(drop=True)], axis=1)
    ABT.parent.mkdir(parents=True, exist_ok=True)
    abt.to_csv(ABT, index=False)
    print(f"\n[6/6][OK] ABT salva em: {ABT}")
    print(f"          {abt.shape[0]:,} linhas x {abt.shape[1]} colunas")
    print(f"          Taxa de inadimplencia preservada: {y.mean():.2%}")


if __name__ == "__main__":
    main()
