# 📊 Modelo de Machine Learning para Definição Inteligente de Limite de Crédito

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

O principal objetivo é desenvolver uma solução baseada em Inteligência Artificial que auxilie na definição do limite de crédito mais adequado para cada cliente, considerando seu perfil financeiro e o risco associado, aproximando a prática acadêmica dos desafios encontrados no mercado de trabalh

---

# 🎯 Problema de Negócio

A definição do limite de crédito é uma etapa estratégica para instituições financeiras, pois impacta diretamente o risco de inadimplência, a rentabilidade e a experiência do cliente.

Conceder o mesmo limite de empréstimo para clientes com perfis financeiros distintos pode resultar tanto em perdas financeiras, ao oferecer crédito acima da capacidade de pagamento, quanto em perda de oportunidades de negócio, ao conceder limites inferiores ao potencial de clientes de baixo risco.

Embora dois clientes possam apresentar um bom histórico de pagamento, suas características financeiras, capacidade de pagamento e perfil de risco podem indicar limites de crédito bastante diferentes. Por exemplo, enquanto um cliente pode ter capacidade para um empréstimo de **R$ 5.000**, outro pode ser elegível para **R$ 40.000**.

Em muitas instituições, essa decisão ainda é baseada em regras fixas e critérios pouco personalizados.

---

# 🧠 Objetivo do Projeto

O objetivo deste projeto é desenvolver uma solução baseada em Machine Learning para auxiliar instituições financeiras na definição do limite de crédito ideal para cada cliente. Utilizando dados históricos e características financeiras, o modelo estima um limite de crédito personalizado, buscando equilibrar a exposição ao risco da instituição, minimizar perdas por inadimplência e ampliar o acesso ao crédito de forma responsável.

---

# 📂 Estrutura do Projeto

```text
.
├── 📁 data/
│   ├── 📁 raw/
│   ├── 📁 processed/
│   └── 📁 external/
│
├── 📁 notebooks/
│
├── 📁 src/
│   ├── 📁 data/
│   ├── 📁 features/
│   ├── 📁 models/
│   ├── 📁 evaluation/
│   └── 📁 utils/
│
├── 📁 models/
├── 📁 reports/
│   ├── 📁 figures/
│   └── 📁 metrics/
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore
```

## 📁 Descrição dos Diretórios

| Diretório | Descrição |
|-----------|-----------|
| 📁 **data/raw** | Dados originais obtidos da fonte, sem qualquer tratamento. |
| 📁 **data/processed** | Dados tratados e prontos para utilização nos modelos. |
| 📁 **data/external** | Bases de dados complementares utilizadas no projeto. |
| 📓 **notebooks** | Notebooks de exploração dos dados, experimentos e análises. |
| 🛠️ **src/data** | Scripts de leitura, limpeza e preparação dos dados. |
| ⚙️ **src/features** | Scripts responsáveis pela engenharia e seleção de atributos. |
| 🤖 **src/models** | Scripts para treinamento, validação e salvamento dos modelos. |
| 📈 **src/evaluation** | Scripts de avaliação e cálculo das métricas dos modelos. |
| 🔧 **src/utils** | Funções auxiliares reutilizadas no projeto. |
| 💾 **models** | Modelos treinados e serializados (.pkl, .joblib, etc.). |
| 📊 **reports** | Relatórios e resultados gerados durante o projeto. |
| 🖼️ **reports/figures** | Gráficos e visualizações. |
| 📑 **reports/metrics** | Métricas e resultados dos experimentos. |

---

# 🗃️ Base de Dados

Descreva a base de dados utilizada.

**Exemplo:**

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
- 🌳 Decision Tree
- 🌲 Random Forest
- 🚀 XGBoost
- ⚡ LightGBM
- 🐱 CatBoost

---

# 📏 Métricas Utilizadas

- ✅ Accuracy
- 🎯 Precision
- 🔎 Recall
- ⚖️ F1-Score
- 📈 ROC-AUC
- 📉 Matriz de Confusão

---

# 🛠️ Tecnologias Utilizadas

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 🤖 Scikit-Learn
- 📊 Matplotlib
- 📈 Seaborn
- 🚀 XGBoost
- ⚡ LightGBM

---

# 🚀 Como Executar o Projeto

## 1️⃣ Clone o repositório

```bash
git clone <url-do-repositorio>

cd nome-do-projeto
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

# 🏋️ Como Treinar o Modelo

Execute:

```bash
python src/models/train.py
```

Caso o treinamento seja realizado em um notebook:

```text
📓 notebooks/model_training.ipynb
```

O modelo treinado será salvo em:

```text
📁 models/
```

---

# 📊 Como Avaliar o Modelo

Execute:

```bash
python src/evaluation/evaluate.py
```

Os resultados serão armazenados em:

```text
📁 reports/metrics/
```

---

# 🏅 Resultados

| Modelo | ROC-AUC | Recall | Precision |
|---------|:------:|:------:|:---------:|
| Regressão Logística | 0.74 | 0.63 | 0.70 |
| Random Forest | 0.79 | 0.71 | 0.74 |
| XGBoost | **0.82** | **0.76** | **0.78** |

---

# ⭐ Melhor Modelo

Descreva o modelo escolhido e o motivo.

**Exemplo:**

O modelo **XGBoost** apresentou o melhor desempenho considerando as métricas ROC-AUC e Recall, sendo selecionado como modelo final por apresentar maior capacidade de identificar clientes inadimplentes.

---

# 👩‍💻 Autores

**Cláudia dos Santos Silva**

**Daniel Azevedo**

**Elizabeth**

📚 Pós Graduação em Engenharia de Dados

🏫 FIA

📅 2025