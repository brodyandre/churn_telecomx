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

## 📚 Conteúdo do Projeto: Desenvolvemos um sumário interativo com os principais estudos realizados, para facilitar a navegação do usuário. Também inserimos um botão de "voltar"
 que permite retornar ao sumário inicial
 
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
```
## 📥 Etapa 3: Carregamento dos Dados

O dataset JSON está em formato plano, o que facilita a leitura direta com pandas.

```bash
df = pd.read_json(url)
df.head()

```
## 🔄 Etapa 5: Normalização dos Dados Aninhados

Expandimos colunas que continham dados no formato JSON aninhado para facilitar a análise.

```bash
df_expandido = pd.concat([
    df_raw.drop(columns=['customer', 'phone', 'internet', 'account']),
    df_raw['customer'].apply(pd.Series),
    df_raw['phone'].apply(pd.Series),
    df_raw['internet'].apply(pd.Series),
    df_raw['account'].apply(pd.Series)
], axis=1)

df_expandido.rename(columns={"customerID": "id_cliente"}, inplace=True)
df_expandido.head()

```
## 📦 Etapa 6: Expansão da Coluna Charges

Separação dos valores financeiro mensais e totais para análises detalhadas.

```bash
df_expandido = pd.concat([
    df_expandido.drop(columns=['Charges']),
    df_expandido['Charges'].apply(pd.Series)
], axis=1)

df_expandido.rename(columns={'Monthly': 'valor_mensal', 'Total': 'valor_total'}, inplace=True)
df_expandido['valor_mensal'] = pd.to_numeric(df_expandido['valor_mensal'], errors='coerce').fillna(0)
df_expandido['valor_total'] = pd.to_numeric(df_expandido['valor_total'], errors='coerce').fillna(0)

df_expandido.head()

```
## 🚀 Etapa 7: Análise Exploratória Visual do Churn

Visualizações interativas para identificar padrões e comportamentos relacionados ao churn.

```bash
import plotly.express as px

def plot_churn_distribution(df):
    df_filtered = df[df['Churn'].isin(['Yes', 'No'])]
    fig = px.histogram(df_filtered, x='Churn', color='Churn',
                       title='Distribuição de Churn - Telecom X',
                       color_discrete_map={'Yes':'red', 'No':'green'})
    fig.show()

```

📌 O que fizemos:
* Filtramos os dados para considerar apenas os valores válidos na coluna Churn (ou seja, 'Yes' ou 'No').

* Criamos um gráfico interativo de barras usando Plotly Express para visualizar o número de clientes que cancelaram ou permaneceram.

* Adicionamos cores distintas para facilitar a leitura (vermelho para churn e verde para permanência).

* Incluímos uma linha de referência para destacar um limiar mínimo de churn (personalizável).

* Mostramos anotações com os percentuais de cada grupo diretamente sobre as barras.

🛠️ Função criada:

```bash
plot_churn_distribution(df, min_threshold=0.15)

```
Esta função modulariza a geração do gráfico, permitindo reutilização com diferentes limiares. Também evita mutações acidentais do DataFrame original ao utilizar uma cópia.

## 🎯 Etapa 8: Gráfico 2 - Distribuição de Churn por Tipo de Contrato

Compreender como o tipo de contrato impacta no churn é essencial para desenhar estratégias de retenção.

📊 Objetivo:
Avaliar se há um padrão entre o tipo de contrato (Mensal, Anual, Bianual) e a probabilidade de cancelamento.

🧩 Insights Esperados:
* Contratos mensais tendem a apresentar churn mais elevado?

* Contratos de longo prazo garantem maior fidelização?

📈 Detalhes do gráfico:
Tipo: Histograma agrupado (barmode='group')

* Eixo X: Tipo de contrato (Contract)

* Eixo Y: Número de clientes

* Cores: 'Yes' em vermelho e 'No' em verde

Este gráfico ajuda a identificar possíveis fragilidades nos planos mensais, além de embasar decisões de políticas de incentivo para contratos mais longos.

## 📊 Etapa 9: Gráfico 3 - Cancelamentos por Método de Pagamento
Aqui buscamos entender se há alguma relação entre o método de pagamento e o churn.

💡 Hipóteses investigadas:
O método "Electronic check" apresenta churn maior?

* Clientes que utilizam pagamentos tradicionais (como boleto ou correio) são mais fiéis?

* Algum método claramente contribui para maior retenção?

📈 Detalhes do gráfico:
Eixo X: PaymentMethod

* Agrupamento: Por Churn ('Yes' e 'No')

* Cores padronizadas: Vermelho para cancelamento, verde para retenção

* Rótulos e eixo com rotação para facilitar a leitura

Este gráfico é essencial para equipes de produto e finanças que buscam oferecer opções de pagamento mais eficazes.

## 🌐 Etapa 10: Gráfico 4 - Cancelamentos por Tipo de Internet
Serviços de internet impactam diretamente a experiência do cliente. A qualidade e disponibilidade da conexão podem influenciar na decisão de cancelar.

🔍 O que analisamos:
Fibra Óptica está associada a maior churn?

* Clientes sem internet são mais propensos a permanecer?

* O serviço DSL se mostra mais estável?

📈 Detalhes do gráfico:
Eixo X: InternetService

* Agrupamento: Por churn

* Legenda: Tipos de internet (DSL, Fiber optic, None)

* Esse gráfico fornece insights sobre a relação entre infraestrutura e churn, sendo particularmente útil para decisões de investimento em tecnologia e rede.

✅ Conclusão parcial:

* As etapas acima revelam tendências valiosas que podem ser exploradas para diminuir o churn e aumentar a retenção de clientes da Telecom X. Com essas visualizações, conseguimos:

* Detectar segmentos de risco

* Apontar oportunidades de retenção

* Guiar decisões estratégicas de negócio

## 📞 Etapa 11: Churn por Serviço de Telefonia
Este gráfico investiga se a presença do serviço de telefonia está associada ao cancelamento dos clientes.

```bash
import plotly.express as px

fig_phone = px.histogram(
    df_expandido,
    x='PhoneService',
    color='Churn',
    barmode='group',
    text_auto=True,
    title='📞 Cancelamento por Serviço de Telefonia',
    labels={'PhoneService': 'Possui Serviço Telefônico', 'Churn': 'Cancelamento'}
)

fig_phone.update_layout(
    xaxis_title='Serviço de Telefonia',
    yaxis_title='Número de Clientes',
    legend_title='Cancelamento',
    bargap=0.3,
    template='plotly_white',
    title_font_size=22
)

fig_phone.show()

```
📊 Insight:
* A maioria dos clientes utiliza o serviço de telefonia.

* Analisar a diferença na taxa de churn entre quem possui e quem não possui o serviço ajuda a identificar a relevância desse produto na decisão de cancelamento.

## 💳 Etapa 12: Churn por Forma de Pagamento
Analisamos se há relação entre a forma de pagamento escolhida pelo cliente e sua propensão a cancelar o serviço.

```bash

fig_pagamento = px.histogram(
    df_expandido,
    x='PaymentMethod',
    color='Churn',
    barmode='group',
    text_auto=True,
    title='💳 Cancelamento por Forma de Pagamento',
    labels={'PaymentMethod': 'Forma de Pagamento', 'Churn': 'Cancelamento'}
)

fig_pagamento.update_layout(
    xaxis_title='Forma de Pagamento',
    yaxis_title='Número de Clientes',
    legend_title='Cancelamento',
    bargap=0.3,
    template='plotly_white',
    title_font_size=22
)

fig_pagamento.show()

```
📊 Insight:
* Clientes que usam Electronic Check apresentam maior taxa de cancelamento.

* Isso pode indicar menor fidelização ou maior vulnerabilidade desses clientes.

## 📡 Etapa 13: Churn por Tipo de Internet
Aqui, verificamos se o tipo de serviço de internet contratado está relacionado ao churn.

```bash
fig_internet = px.histogram(
    df_expandido,
    x='InternetService',
    color='Churn',
    barmode='group',
    text_auto=True,
    title='📡 Cancelamento por Tipo de Internet',
    labels={'InternetService': 'Tipo de Internet', 'Churn': 'Cancelamento'}
)

fig_internet.update_layout(
    xaxis_title='Tipo de Internet',
    yaxis_title='Número de Clientes',
    legend_title='Cancelamento',
    bargap=0.3,
    template='plotly_white',
    title_font_size=22
)

fig_internet.show()

```
📊 Insight:
* Clientes com Fiber Optic cancelam mais do que os com DSL ou sem internet.

* Pode haver problemas de custo ou performance percebida com esse tipo de serviço.

## 💰 Etapa 14: Distribuição dos Valores Mensais por Churn
Avaliamos se há relação entre o valor mensal pago pelo cliente e sua decisão de cancelar o serviço.

```bash
df_filtrado = df_expandido[df_expandido['Churn'].isin(['Yes', 'No'])]

fig_valor_mensal = px.box(
    df_filtrado,
    x='Churn',
    y='valor_mensal',
    color='Churn',
    title='💰 Distribuição do Valor Mensal por Cancelamento',
    labels={'Churn': 'Cancelamento', 'valor_mensal': 'Valor Mensal (R$)'}
)

fig_valor_mensal.update_layout(
    xaxis_title='Cancelamento',
    yaxis_title='Valor Mensal (R$)',
    showlegend=False,
    template='plotly_white',
    title_font_size=22
)

fig_valor_mensal.show()

```
📊 Insight:
* Clientes que cancelaram pagam valores mais altos, em média.

* Isso pode indicar sensibilidade ao preço ou percepção negativa do custo-benefício.

## ⏳ Etapa 15: Tempo de Permanência e Churn
Neste gráfico de boxplot, analisamos o tempo de permanência dos clientes que cancelaram e dos que permaneceram.

```bash
df_plot = df_expandido[df_expandido['Churn'].isin(['Yes', 'No'])]

fig_tenure = px.box(
    df_plot,
    x='Churn',
    y='tenure',
    color='Churn',
    title='⏳ Tempo de Permanência dos Clientes por Status de Cancelamento',
    labels={
        'Churn': 'Cancelamento',
        'tenure': 'Tempo de Permanência (meses)'
    }
)

fig_tenure.update_layout(
    template='plotly_white',
    title_font_size=22,
    showlegend=False
)

fig_tenure.show()

```
📊 Insight:
* Clientes que cancelam tendem a ficar menos tempo com a empresa.

* A mediana de tempo é mais baixa entre clientes que saem, sugerindo foco em retenção nos primeiros meses.

## 📊 Etapa 16: Gráfico — Taxa de Churn por Tipo de Contrato
Essa visualização nos ajuda a entender o impacto que o tipo de contrato tem na retenção de clientes da Telecom X.

🧮 Cálculo
A taxa de churn foi calculada da seguinte forma:

1. Agrupamento dos dados por tipo de contrato e status de churn.

2. Cálculo da porcentagem de clientes que cancelaram em relação ao total de cada tipo de contrato.

3. Tradução opcional dos nomes dos contratos para português.

```bash
# Agrupamento e cálculo da taxa de churn
taxa_churn_contrato = (
    df_expandido.groupby(['Contract', 'Churn'])
    .size()
    .reset_index(name='Quantidade')
)

total_por_contrato = taxa_churn_contrato.groupby('Contract')['Quantidade'].transform('sum')
taxa_churn_contrato['Percentual'] = taxa_churn_contrato['Quantidade'] / total_por_contrato * 100

# Tradução para português
taxa_churn_contrato['Contrato'] = taxa_churn_contrato['Contract'].replace({
    'Month-to-month': 'Mensal',
    'One year': 'Anual (1 ano)',
    'Two year': 'Anual (2 anos)'
})

```
📈 Visualização
Utilizamos Plotly Express para construir um gráfico de barras empilhadas:

```bash
fig_churn_contrato = px.bar(
    taxa_churn_contrato,
    x='Contrato',
    y='Percentual',
    color='Churn',
    color_discrete_map={'No': 'green', 'Yes': 'red'},
    labels={
        'Contrato': 'Tipo de Contrato',
        'Percentual': 'Percentual (%)',
        'Churn': 'Churn (Cancelamento)'
    },
    title='📉 Taxa de Churn por Tipo de Contrato na Telecom X',
    text=taxa_churn_contrato['Percentual'].apply(lambda x: f'{x:.1f}%')
)
fig_churn_contrato.update_layout(barmode='stack', yaxis=dict(ticksuffix='%'))
fig_churn_contrato.show()

```
🧐 Interpretação
* Contrato Mensal apresenta a maior taxa de churn, sugerindo pouca fidelização.

* Contratos de 1 e 2 anos possuem churn significativamente menor.

* Recomendação: Desenvolver estratégias para migrar clientes de contratos mensais para anuais.

## 📅 Etapa 17: Gráfico — Taxa de Churn por Tempo de Contrato (Tenure)
Este gráfico mostra a relação entre o tempo de permanência (tenure) e o churn.

🧮 Cálculo
Convertendo o campo de churn em valores numéricos para calcular a média:

```bash
df_expandido['Churn_num'] = df_expandido['Churn'].map({'Yes': 1, 'No': 0})
churn_por_tenure = df_expandido.groupby('tenure')['Churn_num'].mean().reset_index()

```
📈 Visualização

```bah
fig_tenure = px.line(
    churn_por_tenure,
    x='tenure',
    y='Churn_num',
    labels={
        'tenure': 'Tempo de Contrato (meses)',
        'Churn_num': 'Taxa de Churn'
    },
    title='📅 Taxa de Churn por Tempo de Contrato',
    markers=True
)
fig_tenure.update_layout(yaxis_tickformat=".0%")
fig_tenure.show()

```
🧐 Interpretação
* A taxa de churn é alta nos primeiros meses, o que indica baixa fidelização inicial.

* Após o tempo inicial, a taxa de churn diminui gradualmente.

* Recomendação: Implementar ações de retenção logo após a adesão, como onboarding eficiente, promoções e suporte dedicado.

## 🖼 Etapa 18: Exportação de Gráficos como Imagens (PNG)
Instalamos e testamos o Kaleido, uma engine para exportação de imagens com Plotly:

```bash
!pip install -U kaleido

```
🧪 Teste

```bash
import plotly.express as px
import plotly.io as pio

fig = px.bar(x=["A", "B", "C"], y=[1, 3, 2])
pio.write_image(fig, "/content/test_kaleido.png")

```
💾 Exportação em lote

```bash
figuras_para_salvar = [
    'fig_churn', 'fig_contrato', 'fig_pagamento', 'fig_internet',
    'fig_phone', 'fig_valor_mensal', 'fig_tenure', 'fig_churn_contrato', 'fig'
]

for nome_fig in figuras_para_salvar:
    fig = globals().get(nome_fig)
    if fig is not None:
        caminho = f"/content/{nome_fig}.png"
        fig.write_image(caminho)
        print(f"✅ {nome_fig} salva em {caminho}")
    else:
        print(f"⚠️ Figura {nome_fig} não encontrada.")

```
## 🧾 Etapa 19: Análise Final de Churn — Telecom X
🔍 Visão Geral
Clientes que cancelaram (Churn = Yes) versus os que permaneceram (Churn = No).

📊 Análises Realizadas
* Tipo de Contrato: Mensal tem maior churn.

* Método de Pagamento: Eletrônico associado a maior churn.

* Tecnologia de Internet: Fibra ótica apresenta maior churn.

* Tempo de Contrato: Churn alto nos primeiros meses.

💡 Recomendações Estratégicas
* Retenção Proativa: Focar nos primeiros meses.

* Melhoria na Fibra: Avaliar causas de insatisfação.

* Revisar Formas de Pagamento: Propor alternativas mais engajadoras.

* Ofertas de Fidelização: Incentivar contratos mais longos com vantagens.

✅ Observações Finais
Este relatório foi gerado automaticamente com Python no Google Colab utilizando bibliotecas como pandas, plotly e kaleido.
Todos os gráficos foram salvos em formato .png para possível uso em dashboards ou apresentações.

## 📥 Etapa 20: Download do Relatório Final
Nesta etapa, disponibilizamos para download o relatório analítico em PDF, que resume os principais insights obtidos durante a análise exploratória de dados (EDA) sobre o churn na empresa Telecom X.

* O relatório inclui:

* Gráficos explicativos;

* Estatísticas descritivas;

* Perfis de clientes com maior probabilidade de cancelamento;

* Recomendações estratégicas para redução do churn.

```bash
from IPython.display import FileLink

# Link para download do relatório em PDF
FileLink('/content/relatorio_churn_telecomx.pdf')

```
📎 Clique aqui para baixar o relatório

## 🔜 Etapa 21: Pré-processamento para Machine Learning
Nesta etapa, preparamos o dataset df_expandido para a aplicação de algoritmos de Machine Learning, com foco no Random Forest Classifier, a fim de prever quais clientes possuem maior propensão ao churn.

✅ Etapas Realizadas:
1. Criação da variável alvo binária (Churn_num)
A variável categórica Churn foi convertida para 0 (não cancelou) e 1 (cancelou), compatível com modelos supervisionados.

2.Tratamento de valores nulos
Linhas com dados ausentes nas colunas valor_total ou Churn foram removidas para evitar vieses no treinamento.

3. Separação de features e target
Variáveis como id_cliente foram descartadas, mantendo apenas os atributos relevantes para predição.

4. Codificação de variáveis categóricas
Utilizamos LabelEncoder inicialmente para transformar variáveis textuais em formato numérico.

5. Divisão do dataset
O conjunto foi dividido em 80% treino e 20% teste, preservando a proporção de churn.

6. Treinamento com Random Forest
Um modelo foi treinado para detectar padrões associados ao cancelamento de serviços.

7. Avaliação do desempenho
Métricas como acurácia, matriz de confusão e relatório de classificação foram geradas para medir a performance.

### Diagnóstico de Dados Ausentes

```bash
total_nans = df_expandido.isna().sum().sum()
print(f"Total de valores NaN no dataframe: {total_nans}")

print("\nQuantidade de NaNs por coluna:")
print(df_expandido.isna().sum())

```
## 🎯 Etapa 22: Classificação com Random Forest + SMOTE
* Nesta etapa, aprimoramos o modelo de classificação aplicando duas estratégias importantes:

* One-Hot Encoding para variáveis categóricas;

* SMOTE (Synthetic Minority Oversampling Technique) para balanceamento de classes.

🔁 Pipeline Executado:
1. Conversão da variável Churn para binário

2. Remoção de registros com alvo nulo

3. Separação de X (features) e y (target)

4. Identificação de colunas numéricas e categóricas

5. Aplicação de OneHotEncoder via ColumnTransformer

6. Divisão do dataset em treino/teste com stratify

7. Aplicação de SMOTE no conjunto de treino

8. Treinamento com GridSearchCV e RandomForestClassifier

* Métrica: F1-score

* Validação cruzada: 5-fold

🧠 Melhores hiperparâmetros encontrados:

```bash
{
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5
}

```
📈 Avaliação do Modelo:
Acurácia

* Relatório de Classificação (Precision, Recall, F1)

* Matriz de Confusão

💾 Exportação dos Arquivos:

```bash
joblib.dump(modelo_rf, 'modelo_rf.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')

```
### Downloads automáticos (se no Google Colab):

```bah
from google.colab import files
files.download('modelo_rf.joblib')
files.download('preprocessor.joblib')

```



