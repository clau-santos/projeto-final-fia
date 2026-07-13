# 📊 Modelo de Machine Learning de Sistema Inteligente de Limite de Credito

## 📖 Descrição do Projeto

Este projeto foi desenvolvido como trabalho final da disciplina de **Inteligência Artificial** e tem como objetivo simular o ciclo completo de desenvolvimento de um projeto de Machine Learning dentro de uma organização.

Mais do que construir um modelo preditivo, a proposta é aplicar as etapas que fazem parte de um projeto real de Ciência de Dados, conectando o problema de negócio às soluções baseadas em dados.

Durante o desenvolvimento são abordadas etapas como:

- 📌 Compreensão do problema de negócio;
- 📊 Análise exploratória e preparação dos dados;
- 🧹 Limpeza e transformação das informações;
- 🏗️ Engenharia de atributos;
- 🤖 Treinamento e comparação de modelos de Machine Learning;
- 📈 Avaliação dos resultados por meio de métricas adequadas;
- 💡 Interpretação dos resultados para apoiar a tomada de decisão.

O principal objetivo é desenvolver uma solução baseada em Inteligência Artificial que auxilie na definição do limite de crédito mais adequado para cada cliente, considerando seu perfil financeiro e o risco associado, aproximando a prática acadêmica dos desafios encontrados no mercado de trabalho.

---

# 🎯 Problema de Negócio

A definição do limite de crédito é uma etapa estratégica para instituições financeiras, pois impacta diretamente o risco de inadimplência, a rentabilidade e a experiência do cliente.

Conceder o mesmo limite de empréstimo para clientes com perfis financeiros distintos pode resultar tanto em perdas financeiras, ao oferecer crédito acima da capacidade de pagamento, quanto em perda de oportunidades de negócio, ao conceder limites inferiores ao potencial de clientes de baixo risco.

Embora dois clientes possam apresentar um bom histórico de pagamento, suas características financeiras, capacidade de pagamento e perfil de risco podem indicar limites de crédito bastante diferentes. 
Por exemplo, enquanto um cliente pode ter capacidade para um empréstimo de `R$ 5.000`, outro pode ser elegível para `R$ 40.000`.

Em muitas instituições, essa decisão ainda é baseada em regras fixas e critérios pouco personalizados.

---

# 🧠 Objetivo do Projeto

O objetivo deste projeto é desenvolver uma solução baseada em Machine Learning para auxiliar instituições financeiras na definição do limite de crédito ideal para cada cliente. Utilizando dados históricos e características financeiras, o modelo estima um limite de crédito personalizado, buscando equilibrar a exposição ao risco da instituição, minimizar perdas por inadimplência e ampliar o acesso ao crédito de forma responsável.

---

# 📂 Estrutura do Projeto

```text
.
├── 📁 Dados/
│   ├── 📄 raw.csv
│   ├── 📄 clean_data.csv
│   └── 📄 abt.csv
│
├── 📁 DataPipeline/
│   ├── 📄 data_sanitization.py
│   ├── 📄 abt_transform.py
│   ├── 📄 config.yaml
│
├── 📁 DataValidator/
│   ├── 📄 validator.py

├── 📁 model/
│   ├── 📄 train.py
│   ├── 📄 evaluation.ipynb
│   ├── 📄 config.yaml
│   ├── 📄 credit_policy.py

├── 📄 exp_analysis.ipynb
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore
```

## 📁 Descrição dos Diretórios

| Diretório/Arquivo                   | Descrição                                                         |
|-------------------------------------|-------------------------------------------------------------------|
| 📁 **Dados/raw.csv**                | Dados originais obtidos da fonte, sem qualquer tratamento.        |
| 📁 **Dados/clean_data.csv**         | Dados tratados e prontos para geração da ABT.                     |
| 📁 **Dados/abt.csv**                | Dados tratados e prontos para utilização nos modelos.             |
| 🛠️ **DataPipeline/data_sanitization.py** | Scripts de leitura, limpeza e preparação dos dados.               |
| 📑 **DataPipeline/abt_transform.py** | Scripts responsáveis pela engenharia e seleção de atributos.      |
| 📊 **DataPipeline/exp_analysis**    | Notebooks de exploração dos dados, experimentos e análises.       |
| ⚙️ **DataPipeline/config.yaml**     | Arquivo de configuração para os arquivos em DataPipeline.         |
| 🔧 **DataValidator/validator.py**   | Classes do Pydantic e validação dos contratos de dados esperados. |
| 💾 **Model/train.py**               | Script para treinamento, validação e salvamento dos modelos.      |
| 📓 **Model/evaluation.ipynb**       | Notebooks de exploração dos dados, experimentos e análises.       |
| 🤖 **Model/credit_policy.py**       | Script com a politica de crédito que será utilizada.              |
| ⚙️ **Model/config.yaml**            | Arquivo de configuração do modelo.                                |
| 📑 **requirements.txt**             | Métricas e resultados dos experimentos.                           |
---


## 🔄 Fluxo dos Dados

O projeto segue um pipeline estruturado para transformar os dados brutos em uma recomendação de crédito baseada em Machine Learning.

```text
application_train.csv (Kaggle)
              │
              ▼
1_data_sanitization.py
- Validação dos dados (Pydantic)
- Limpeza e padronização
- Tratamento de valores ausentes
- Remoção de inconsistências
              │
              ▼
clean_data.csv
              │
              ▼
2_abt_transform.py
- Engenharia de atributos
- Seleção de variáveis
- Encoding das variáveis categóricas
- Imputação de valores faltantes
              │
              ▼
abt.csv
              │
              ▼
3_train.py
- Divisão treino/teste
- Treinamento dos modelos
- Avaliação (ROC-AUC, KS e Gini)
- Seleção do melhor modelo
              │
              ▼
model_pd.pkl
metrics.json
test_predictions.csv
              │
              ▼
credit_policy.py
- Estima a Probabilidade de Default (PD)
- Classifica o cliente por faixa de risco
- Calcula o limite de crédito recomendado
- Define o prazo máximo da operação
              │
              ▼
Decisão Final de Crédito
(Aprovar, Limite e Prazo)
```

# 🗃️ Base de Dados

O projeto utiliza a base pública Home Credit Default Risk, disponibilizada no Kaggle.
A base contém aproximadamente 307 mil solicitações de crédito com informações cadastrais, financeiras e comportamentais dos clientes.
O objetivo é prever a probabilidade de inadimplência (TARGET), permitindo apoiar a definição do limite de crédito adequado para novos clientes.

A base contém informações como:

- 💰 Renda
- 💳 Cartões de crédito
- 🏦 Empréstimos anteriores
- 📅 Histórico de pagamentos
- 📈 Parcelas
- 👤 Informações cadastrais

---

# 🔬 Metodologia

O projeto foi desenvolvido seguindo as seguintes etapas:

- 📥 Coleta dos dados
- 🔍 Análise Exploratória dos Dados (EDA)
- 🧹 Limpeza e tratamento dos dados
- ❓ Tratamento de valores ausentes
- 🏗️ Engenharia de atributos
- 🔄 Transformação das variáveis
- ✂️ Divisão entre treino e teste
- 🤖 Treinamento dos modelos
- 🎛️ Ajuste de hiperparâmetros
- 📊 Avaliação dos resultados
- 🏆 Seleção do melhor modelo

---

# 🤖 Modelos Avaliados

- 📉 Regressão Logística
- 🌲 HistGradientBoosting (Modelo Principal)

---

# 📏 Métricas Utilizadas

- 📈 ROC-AUC 
- 📏 KS (Kolmogorov-Smirnov) 
- 📊 Gini 
- 🧩 Matriz de Confusão 
- 🎯 Precision 
- 🔎 Recall 
- ⚖️ F1-Score
---

# 🛠️ Tecnologias Utilizadas

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 🤖 Scikit-Learn
- 📊 Matplotlib
- 📄 PyYAML
- 💾 Joblib
- ✅ Pydantic


---

# 🚀 Como Executar o Projeto

## 1️⃣ Clone o repositório

```bash
git clone <url-do-repositorio>
cd projeto-final-fia
```

---

## 2️⃣ Crie um ambiente virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

---

# 📊 Gerando a ABT

## 1️⃣ Crie a pasta do dataset

Na raiz do projeto, crie uma pasta chamada:

```text
home-credit-default-risk
```

---

## 2️⃣ Baixe o dataset

Faça o download do arquivo:

```text
application_train.csv
```

Disponível na competição **Home Credit Default Risk** do Kaggle.

---

## 3️⃣ Coloque o arquivo na pasta criada

A estrutura deverá ficar assim:

```text
projeto-final-fia/
│
├── home-credit-default-risk/
│   └── application_train.csv
│
├── DataPipeline/
├── Model/
├── Dados/
└── ...
```

---

## 4️⃣ Execute a limpeza dos dados

```bash
python -m DataPipeline/data_sanitization.py
```

---

## 5️⃣ Gere a ABT (Analytical Base Table)

```bash
python -m DataPipeline/abt_transform.py
```

Ao final da execução, a ABT será gerada na pasta de saída do projeto e estará pronta para utilização na etapa de treinamento do modelo.
# 🏋️ Como Treinar o Modelo

Execute:

```bash
python -m Model/train.py
```

O modelo treinado será salvo em:

```text
📁 Model/
```

---

# 📊 Como Avaliar o Modelo

Rode o arquivo:

```text
📓 Model/4_evaluation.ipynb
```

---

# 🏅 Resultados


| Modelo               | ROC-AUC    | KS         | Gini       |
| -------------------- | ---------- | ---------- | ---------- |
| Logistic Regression  | 0.7501     | 0.3707     | 0.5002     |
| HistGradientBoosting | **0.7659** | **0.3976** | **0.5319** |

---

# ⭐ Melhor Modelo

O modelo HistGradientBoosting apresentou o melhor desempenho, alcançando:

- ROC-AUC: 0.7659
- KS: 0.3976
- Gini: 0.5319

Por esse motivo foi selecionado como modelo final para estimar a Probabilidade de Default (PD), utilizada posteriormente pela política de concessão de crédito.

---

# 👩‍💻 Autores

**Cláudia dos Santos Silva**

**Daniel Azevedo**

**Elizabeth Oliveira**

📚 Pós Graduação em Engenharia de Dados

🏫 FIA

📅 2025