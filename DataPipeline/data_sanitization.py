# -*- coding: utf-8 -*-
"""
data_sanitization.py  —  Limpeza e padronizacao dos dados brutos.
Projeto Final FIA/LABDATA · Home Credit Default Risk · Ponto iii

Entrada : application_train.csv (base bruta)
Saida   : Dados/clean_data.csv (base limpa, mesmas colunas, sem inconsistencias)

O que faz (decisoes vindas da EDA do Ponto ii):
  1. Converte o sentinela DAYS_EMPLOYED == 365243 (~1000 anos) em NaN
     e cria a flag FLAG_DAYS_EMPLOYED_ANOM para nao perder a informacao.
  2. Trata categorias invalidas 'XNA' (CODE_GENDER, ORGANIZATION_TYPE) como ausentes.
  3. Remove espacos em branco e padroniza texto das colunas categoricas.
  4. Trata renda absurda (outlier) acima do teto definido no config.
  5. Remove linhas duplicadas pela chave do cliente.

Uso:
  python data_sanitization.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

from DataValidator.validator import data_validator

# ----- localizar raiz do projeto e carregar config -----
ROOT = Path(__file__).resolve().parent.parent          # pasta 'Projeto Final - TCC'
CFG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW   = ROOT / CFG["paths"]["raw"]
CLEAN = ROOT / CFG["paths"]["clean"]
ID    = CFG["target_columns"]["id"]
SAN   = CFG["sanitization"]


def _read_data() -> pd.DataFrame:
    """
    Função responsável por ler o arquivo raw
    """
    if not RAW.exists():
        sys.exit(f"[ERRO] Base bruta nao encontrada: {RAW}\n"
                 f"Ajuste 'paths.raw' no config.yaml.")

    print(f"[1/8] Lendo base bruta: {RAW.name}")
    df = pd.read_csv(RAW)
    return df

def _data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função responsável por realizar a limpeza/tratamento dos dados.
    """
    # 2. Sentinela de DAYS_EMPLOYED -> NaN (+ flag)
    sent = SAN["days_employed_sentinel"]
    mask = df["DAYS_EMPLOYED"] == sent
    df["FLAG_DAYS_EMPLOYED_ANOM"] = mask.astype("int8")
    df.loc[mask, "DAYS_EMPLOYED"] = np.nan
    print(f"[4/8] DAYS_EMPLOYED sentinela ({sent}) -> NaN em {int(mask.sum()):,} linhas")

    # 3. Categorias 'XNA' -> NaN
    for col in SAN["xna_to_nan"]:
        if col in df.columns:
            n = int((df[col] == "XNA").sum())
            df[col] = df[col].replace("XNA", np.nan)
            print(f"[3/6] {col}: 'XNA' -> NaN em {n:,} linhas")

    # 4. Padronizar texto das categoricas (trim + sem duplos espacos)
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip().replace(r"\s+", " ", regex=True)
    print(f"[5/8] Texto padronizado em {len(obj_cols)} colunas categoricas")

    # 5. Renda absurda (outlier) -> NaN
    cap = SAN["income_cap"]
    n_out = int((df["AMT_INCOME_TOTAL"] > cap).sum())
    df.loc[df["AMT_INCOME_TOTAL"] > cap, "AMT_INCOME_TOTAL"] = np.nan
    print(f"[6/8] AMT_INCOME_TOTAL > {cap:,} -> NaN em {n_out:,} linhas")

    # 6. Remover duplicatas pela chave
    dup = int(df.duplicated(subset=[ID]).sum())
    df = df.drop_duplicates(subset=[ID])
    print(f"[7/8] Duplicatas por {ID} removidas: {dup:,}")
    return df


def _data_validation(df: pd.DataFrame, model: str) -> None:
    """
    Funcão responsável por realizar a validação do contrato de dados
    """
    records = df.replace({np.nan: None}).to_dict(orient="records")
    data_validator(data=records, model=model)

def main():
    df = _read_data()

    print(f"[2/8 Iniciando validação do contrato de dados da camada RAW: {RAW.name}")
    _data_validation(df=df, model="RAW")

    print(f"[3/8] Lendo base bruta: {RAW.name}")

    n0, c0 = df.shape
    print(f"      -> {n0:,} linhas x {c0} colunas")

    df = _data_transform(df=df)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)

    print(f"[8/8 Iniciando validação do contrato de dados para os dados da camada SILVER.")
    _data_validation(df=df, model="SILVER")

    df.to_csv(CLEAN, index=False)
    print(f"\n[OK] clean_data salvo em: {CLEAN}")
    print(f"     {df.shape[0]:,} linhas x {df.shape[1]} colunas")


if __name__ == "__main__":
    main()
