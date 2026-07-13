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

CLEAN_DATA_PATH = ROOT / CFG["paths"]["clean"]
ABT_PATH = ROOT / CFG["paths"]["abt"]
TARGET_COLUMNS = CFG["target_columns"]
ABT_PARAMETERS = CFG["abt"]
COLUMNS_TO_DROP = CFG["columns_to_drop"]


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

def read_clean_data():
    if not CLEAN_DATA_PATH.exists():
        sys.exit(f"[ERRO] clean_data nao encontrado: {CLEAN_DATA_PATH}\n"
                 f"Rode antes: python data_sanitization.py")

    print(f"[1/7] Lendo base limpa: {CLEAN_DATA_PATH.name}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    return df


def abt_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # guardar id e alvo, trabalhar nas features
    y  = df[TARGET_COLUMNS["target"]]
    ids = df[TARGET_COLUMNS["id"]]

    keep_columns = [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS",
        "AMT_ANNUITY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "NAME_INCOME_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_EDUCATION_TYPE",
        TARGET_COLUMNS["target"],
        TARGET_COLUMNS["id"]
    ]

    df = df[keep_columns]

    model_data  = df.drop(columns=[TARGET_COLUMNS["id"], TARGET_COLUMNS["target"]])

    # 1. Descartar colunas muito vazias
    thr = ABT_PARAMETERS["null_drop_threshold"]
    null_ratio = model_data.isna().mean()
    if null_ratio.isna().any():
        drop_cols = null_ratio[null_ratio > thr].index.tolist()
        model_data = model_data.drop(columns=drop_cols)
        print(f"[2/7] Colunas descartadas por >{thr:.0%} nulos: {len(drop_cols)}")

    # 2. Engenharia de atributos
    model_data = engenharia_atributos(model_data)
    print(f"[3/7] Features derivadas criadas (idade, razoes, EXT_SOURCE_*)")

    # 3. Delete de colunas que não são necessárias
    model_data = model_data.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    print(f"[4/7] Colunas descartadas por serem desnecessárias: {len(COLUMNS_TO_DROP)}")

    # 4. Encoding de categoricas
    cat_cols = model_data.select_dtypes(include="object").columns.tolist()
    freq_cols = [c for c in ABT_PARAMETERS["freq_encode_cols"] if c in cat_cols]
    # 3a. frequency encoding (alta cardinalidade)
    for c in freq_cols:
        freq = model_data[c].value_counts(normalize=True)
        model_data[c + "_FREQ"] = model_data[c].map(freq).fillna(0.0)
    model_data = model_data.drop(columns=freq_cols)
    # 3b. one-hot nas demais (dentro do limite de cardinalidade)
    onehot_cols = [c for c in cat_cols if c not in freq_cols
                   and model_data[c].nunique(dropna=True) <= ABT_PARAMETERS["onehot_max_cardinality"]]
    model_data = pd.get_dummies(model_data, columns=onehot_cols, dummy_na=True, dtype="int8")
    # qualquer categorica restante (acima do limite e nao listada) -> frequency
    resto = model_data.select_dtypes(include="object").columns.tolist()
    for c in resto:
        freq = model_data[c].value_counts(normalize=True)
        model_data[c + "_FREQ"] = model_data[c].map(freq).fillna(0.0)
    if resto:
        model_data = model_data.drop(columns=resto)
    print(f"[5/7] Encoding: {len(freq_cols)+len(resto)} frequency + {len(onehot_cols)} one-hot")

    # 5. Imputacao numerica pela mediana
    num_cols = model_data.select_dtypes(include=[np.number]).columns
    model_data[num_cols] = model_data[num_cols].fillna(model_data[num_cols].median())
    # substituir eventuais infinitos por 0 (razoes com denominador minimo)
    model_data = model_data.replace([np.inf, -np.inf], 0.0)
    print(f"[6/7] Imputacao mediana aplicada; nulos restantes: {int(model_data.isna().sum().sum())}")

    # cast para float32 (opcional, reduz tamanho)
    if ABT_PARAMETERS.get("cast_float32", False):
        fcols = model_data.select_dtypes(include=["float64"]).columns
        model_data[fcols] = model_data[fcols].astype("float32")

    # 6. Montar ABT final = ID + TARGET + features
    abt = pd.concat([ids.reset_index(drop=True), y.reset_index(drop=True),
                     model_data.reset_index(drop=True)], axis=1)
    return abt, y

def main():
    df = read_clean_data()
    print(f"      -> {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    abt, y = abt_transform(df=df)
    ABT_PATH.parent.mkdir(parents=True, exist_ok=True)
    abt.to_csv(ABT_PATH, index=False)

    print(f"\n[7/7][OK] ABT salva em: {ABT_PATH}")
    print(f"          {abt.shape[0]:,} linhas x {abt.shape[1]} colunas")
    print(f"          Taxa de inadimplencia preservada: {y.mean():.2%}")


if __name__ == "__main__":
    main()
