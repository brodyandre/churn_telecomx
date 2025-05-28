# 📉 Análise e Previsão de Churn - Telecom X

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-data_analysis-brightgreen?logo=pandas)]
[![Plotly](https://img.shields.io/badge/Plotly-visualization-orange?logo=plotly)]
[![Streamlit](https://img.shields.io/badge/Streamlit-web_app-red?logo=streamlit)]

---

## 🚀 Visão Geral

Este projeto, desenvolvido por **Luiz André** (GitHub: [@brodyandre](https://github.com/brodyandre)), tem como objetivo a análise exploratória e preditiva do *churn* (evasão de clientes) da empresa fictícia **Telecom X**. 

A partir de dados JSON reais, o projeto realiza um estudo completo que envolve desde o carregamento e limpeza dos dados até a criação de modelos de machine learning para previsão de cancelamentos. O resultado final está preparado para integração com uma aplicação web usando **Streamlit** para facilitar a visualização dos insights e suporte à tomada de decisão.

---

## 📚 Conteúdo do Projeto

<a name="indice"></a>

- [📉 Etapa 1: Estudo de Churn - Telecom X](#etapa-1-estudo-de-churn---telecom-x)
- [🚀 Etapa 2: Carregando e Normalizando os Dados JSON](#etapa-2-carregando-e-normalizando-os-dados-json)
- [📥 Etapa 3: Carregamento dos Dados](#etapa-3-carregamento-dos-dados)
- [🔍 Etapa 4: Análise Exploratória dos Dados (EDA)](#etapa-4-análise-exploratória-dos-dados-eda)
- [🔄 Etapa 5: Normalização dos Dados Aninhados](#etapa-5-normalização-dos-dados-aninhados)
- [📦 Etapa 6: Expansão da Coluna Charges](#etapa-6-expansão-da-coluna-charges)
- [🚀 Etapa 7: Análise Exploratória Visual do Churn da Telecom X](#etapa-7-análise-exploratória-visual-do-churn-da-telecom-x)
- [🎯 Etapas 8 a 31: Visualizações e Modelagem Preditiva](#etapa-8-gráfico-2-distribuição-de-churn-por-tipo-de-contrato)
- [📥 Etapa 20: Download do Relatório](#etapa-20-download-do-relatório)
- [🔜 Etapa 21: Próximo Passo: Pré-processamento para Machine Learning](#etapa-21-próximo-passo-pré-processamento-para-machine-learning)

---

## 📉 Etapa 1: Estudo de Churn - Telecom X

Este projeto visa compreender o comportamento dos clientes da Telecom X para identificar padrões de cancelamento (*churn*) e construir modelos preditivos para melhorar a retenção.

### Objetivos:
- Identificar padrões de churn.
- Criar modelos preditivos de alta acurácia.
- Gerar insights para ações estratégicas.

### Fonte dos dados:
[TelecomX_Data.json](https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/refs/heads/main/TelecomX_Data.json)

---

## 🚀 Etapa 2: Carregando e Normalizando os Dados JSON

```python
import pandas as pd

url = "https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/refs/heads/main/TelecomX_Data.json"
df_raw = pd.read_json(url)
df_raw.head()
