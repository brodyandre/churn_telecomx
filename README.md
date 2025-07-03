# 📊 Projeto: Previsão de Churn - Telecom X - com Random Forest e Deploy no Streamlit

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repositório-24292e?style=for-the-badge&logo=github&logoColor=white)](https://github.com/brodyandre/churn_telecomx)
[![Streamlit App](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://churntelecomx-unhnurkqur8sppnwipdczy.streamlit.app/)
[![License](https://img.shields.io/github/license/brodyandre/churn_telecomx?style=for-the-badge)](https://github.com/brodyandre/churn_telecomx/blob/main/LICENSE)

#### para acessar o dashboard interativo, ja com o modelo random forest treinado e devidamente ajustado. Clique no icone do stremalite acima
---

# 🎯 **Objetivo**:  
Este projeto tem como propósito prever a **evasão de clientes (churn)** em uma empresa de telecomunicações fictícia — a **Telecom X** — utilizando técnicas de **aprendizado de máquina**, **análise exploratória de dados**, visualizações interativas com **Streamlit** e boas práticas de projeto em ciência de dados. Este projeto foi criado para o desafio do programa da **Oracle One** ao qual faço parte.

🧠 Desenvolvido com foco em aprendizado, melhoria contínua e contribuição à comunidade científica.

🔗 Acesse o projeto completo aqui:  
👉 **[https://github.com/brodyandre/churn_telecomx](https://github.com/brodyandre/churn_telecomx)**

---

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

## 📚 Conteúdo do Projeto: 
Desenvolvemos um sumário interativo com os principais estudos realizados, para facilitar a navegação do usuário. Também inserimos um botão de "voltar"
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

### Obs: caso aconteça algum erro na celula de instalação do kaleido conforme abaixo:

```bash
# Célula 1: Instalar Kaleido
!pip install -U kaleido

```
Clique no menu **Ambiente de execução** e em seguida **Reiniciar sessão**. Agora ainda no menu **Ambiente de execução** podemos clicar em: **Executar tudo**. Todas as células restantes serão executadas sem mensagens de erro.

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
## 📊 Etapa 23: Importância das Features no Random Forest
Nesta etapa, identificamos quais variáveis foram mais relevantes na predição do churn com base no modelo final ajustado via GridSearch.

🛠️ Procedimentos Realizados:
1. Extração dos scores de importância
Utilizando o atributo .feature_importances_ do melhor modelo.

2. Reconstrução dos nomes das variáveis

* Pegamos os nomes gerados pelo OneHotEncoder.

* Combinamos com os nomes das variáveis numéricas (mantidas diretamente no modelo).

3. Criação de DataFrame de Importância
Um DataFrame foi construído com as variáveis e seus respectivos scores, ordenado de forma decrescente.

4. Visualização Gráfica
Utilizamos um gráfico de barras horizontais para destacar as top 20 variáveis mais relevantes na decisão do modelo.

📌 Esta análise é fundamental para:

* Guiar estratégias de retenção, identificando fatores críticos de churn;

* Fornecer insights de negócio com base nos dados preditivos.

🔜 Na próxima etapa, visualizaremos esse gráfico com o pacote matplotlib ou seaborn, facilitando a comunicação dos resultados com stakeholders.

## 🎯 Etapa 24: Treinamento e Avaliação do Modelo Random Forest com as 10 Features Mais Importantes
Nesta etapa, buscamos reduzir a complexidade do modelo utilizando apenas as 10 variáveis mais relevantes, segundo a análise de importância de features feita com RandomForest. Isso nos ajuda a responder perguntas como:

* É possível manter uma boa performance preditiva com menos variáveis?

* Quais são as variáveis mais impactantes na previsão de churn?

* Reduzindo o número de colunas, o modelo ganha em desempenho e interpretabilidade?

✅ Objetivo:
Avaliar o desempenho de um novo modelo Random Forest treinado apenas com as 10 features mais importantes, comparando seus resultados com o modelo completo.

📌 Principais Passos da Implementação:
1. Extração das Importâncias das Features
Utilizamos o atributo .feature_importances_ do melhor modelo Random Forest (encontrado via GridSearchCV) para extrair a importância relativa de cada variável já transformada pelo pipeline.

2. Seleção das Top 10 Features
Criamos um DataFrame com os nomes das features e seus respectivos pesos, ordenando do mais importante para o menos.

Selecionamos as 10 primeiras features para compor o novo conjunto de dados.

3. Redução das Matrizes de Treino e Teste
Usamos a transformação preprocessor.transform() para obter as versões numéricas de X_train e X_test.

Selecionamos somente as colunas correspondentes às top 10 features.

4. Reaplicação do SMOTE
Reaplicamos o SMOTE somente no conjunto de treino com as features reduzidas para balancear novamente as classes.

5. Treinamento do Novo Modelo
Instanciamos e treinamos um novo RandomForestClassifier utilizando os mesmos hiperparâmetros ótimos encontrados anteriormente, mas com o conjunto reduzido.

6. Avaliação do Modelo com 10 Variáveis
Avaliamos o modelo reduzido usando acurácia, relatório de classificação e matriz de confusão.

📊 Interpretação Esperada
* Reduzir o número de variáveis pode:

* Tornar o modelo mais rápido e leve;

* Aumentar a interpretabilidade dos resultados;

* Reduzir risco de overfitting;

* Revelar quais variáveis realmente fazem diferença no churn.

* No entanto, é essencial validar se o modelo simplificado mantém um desempenho aceitável.

🧠 Boas Práticas
* A análise de features mais importantes pode variar entre algoritmos — esta análise é específica do RandomForestClassifier.

* Sempre valide com o conjunto de teste para garantir que a simplificação não traga perda de desempenho.

* Use esse tipo de abordagem para explicar melhor os resultados a pessoas não técnicas (ex.: áreas de negócios, marketing).

📌 Exemplo de Saída Esperada:

```bash
Top 10 features mais importantes:
         feature     importance
0  tenure_scaled       0.15293
1  MonthlyCharges   0.12983
2  Contract_Two year  0.09823
...

```
📈 Exemplo de Métrica:

```bash
Acurácia com top 10 features: 0.8182

```
🔁 Continue a análise avaliando o impacto de thresholds e curvas ROC/PR na etapa seguinte!


## 🔍 Etapa 30: Avaliação de Métricas em Vários Thresholds e Visualização Gráfica

Nesta etapa, vamos analisar como diferentes valores de **threshold** afetam as métricas de avaliação do modelo de classificação.

Essa análise é fundamental para **problemas de churn**, onde queremos equilibrar **recall** (não perder clientes churners) e **precisão** (não classificar erroneamente clientes fiéis como churn).

A função `avaliar_varios_thresholds` executa essa tarefa e gera:
- Um **DataFrame** com métricas por threshold
- Um **gráfico de linhas** comparando Precision, Recall, F1 Score e Acurácia
- Um arquivo CSV com os resultados

Vamos implementar e visualizar esses dados agora.

### 🧠 Função avaliar_varios_thresholds

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def avaliar_varios_thresholds(modelo, X_test, y_test, thresholds=np.arange(0.1, 1.0, 0.1)):
    resultados = []

    # Probabilidades da classe positiva
    probs = modelo.predict_proba(X_test)[:, 1]

    # Avaliação em diferentes thresholds
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        resultados.append({
            'threshold': thr,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': acc
        })

    # Criar DataFrame com os resultados
    df_resultados = pd.DataFrame(resultados)

    # Salvar CSV
    df_resultados.to_csv('avaliacao_thresholds.csv', index=False)
    print("Resultados salvos em 'avaliacao_thresholds.csv'")

    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultados['threshold'], df_resultados['precision'], marker='o', label='Precision')
    plt.plot(df_resultados['threshold'], df_resultados['recall'], marker='o', label='Recall')
    plt.plot(df_resultados['threshold'], df_resultados['f1_score'], marker='o', label='F1 Score')
    plt.plot(df_resultados['threshold'], df_resultados['accuracy'], marker='o', label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Métricas vs Threshold')
    plt.grid(True)
    plt.legend()
    plt.show()

    return df_resultados

```
### ✅ Execução da função e visualização

```bash
# Avaliação do modelo com diferentes thresholds
df_avaliacao = avaliar_varios_thresholds(grid_search.best_estimator_, X_test_prep, y_test)

# Visualizar o DataFrame resultante
df_avaliacao

```
## 📊 Etapa 31: Relatório de Análise dos Thresholds para Previsão de Churn – TelecomX
✅ Objetivo
Avaliar os resultados obtidos na Etapa 30 e identificar o threshold mais adequado para classificar corretamente os clientes propensos ao cancelamento (churn), visando apoiar a tomada de decisões estratégicas da empresa.

No contexto de churn, damos maior peso ao Recall da classe 1, pois é mais prejudicial deixar de identificar um cliente que irá sair do que cometer um falso positivo.

📌 Critérios de Avaliação
* Recall elevado da classe 1 (churners): fundamental para captar o maior número de clientes que realmente irão cancelar.

* F1-score balanceado: representa o compromisso entre precisão e recall.

* Precision aceitável: evita o desperdício de recursos com clientes que não cancelariam.

* Accuracy contextual: embora importante, pode ser enganosa em datasets desbalanceados.

```bash
### 📊 Tabela Resumo das Métricas por Threshold

| Threshold | Precision | Recall | F1 Score | Accuracy | Comentário |
|-----------|-----------|--------|----------|----------|------------|
| 0.1       | 0.52 🟡    | 🔺 **0.96** | 0.67 🟡  | 0.57 🔻  | Recall muito alto, muitos falsos positivos. |
| 0.2       | 0.58      | 0.91   | 0.71     | 0.63     | Boa sensibilidade, baixa precisão. |
| 0.3       | 0.63      | 0.84   | 0.72     | 0.68     | Equilíbrio razoável. |
| **0.4**   | 🟢 **0.67** | 🟢 **0.74** | ✅ **0.70** | 🟢 **0.71** | 🔹 Melhor F1 Score — ideal para churn |
| 0.5       | 0.72      | 0.67   | 0.69     | 0.75     | Acurácia alta, recall começa a cair. |
| 0.6       | 0.78      | 0.52   | 0.62     | 0.79     | Precision alta, recall baixo. |
| 0.7       | 0.83      | 0.41   | 0.55     | 0.82     | Perde muito recall. |
| 0.8       | 🔵 0.89   | 0.29   | 0.44     | 0.85     | Fraco para churn prediction. |
| 0.9       | 🔵 0.91   | 🔻 **0.18** | 🔻 **0.30** | 0.86 | Ignora quase todos os churners. |

---

✅ **Observações**:

- A linha **0.4** é destacada como melhor threshold com base no F1 Score.


```

🟩 Conclusão: Melhor Threshold
Com base na análise:

Threshold recomendado: 0.4

Métricas associadas:

Precision: 0.53

Recall: 0.77 ✅

F1-score: 0.63

Accuracy: 0.75

### 🔍 Justificativa: Este valor proporciona um ótimo equilíbrio entre detectar clientes que vão cancelar (sensibilidade) e evitar falsos positivos em excesso. Ideal para ações preventivas e estratégias de retenção da equipe da BRA TeleCOM.

### 🎯 Recomendação Estratégica
* Adotar threshold = 0.4 como ponto de decisão para marcar um cliente como churner.

* Criar um grupo de risco intermediário para clientes com probabilidade entre 0.4 e 0.6, que serão tratados com atenção especial em campanhas de retenção personalizadas.

### 🗃️ Arquivos Gerados
* avaliacao_thresholds.csv': contém todas as métricas por threshold.

* Gráfico: exibido na etapa anterior, mostrando as curvas de Precision, Recall, F1 Score e Accuracy.

### 🚀 Próximos Passos
* Implementar o threshold escolhido no modelo de produção.

* Monitorar periodicamente o desempenho do modelo e reavaliar o threshold com novos dados.

* Avaliar impacto de ações de retenção baseadas nesse modelo em KPIs de churn.

## Documentação do Dashboard Interativo de Previsão de Churn - Telecom X (app.py)
Introdução
Este projeto apresenta um dashboard interativo desenvolvido com Streamlit, para análise e previsão de churn (cancelamento de clientes) da empresa Telecom X. O dashboard utiliza um modelo Random Forest previamente treinado e ajustado (tunning) para realizar previsões precisas.

O objetivo é fornecer uma ferramenta visual para a equipe de análise e gerência, permitindo a exploração dos dados com filtros dinâmicos e métricas relevantes.

Tecnologias Utilizadas
* Python 3.x

* Pandas

* Streamlit

Instalação
Para rodar o dashboard localmente, instale as bibliotecas necessárias:

```bash
pip install pandas
pip install streamlit

```

Estrutura do Código
1. Configuração da Página

```bash
st.set_page_config(page_title="Dashboard Churn - Big Insights", layout="wide")

```
Define o título da aba do navegador e a largura do layout do dashboard, melhorando a visualização e usabilidade.

2. Carregamento e Cache dos Dados
```bash
@st.cache_data
def load_data():
    df = pd.read_csv('df_expandido.csv')
    return df

df = load_data()

```
* A função load_data carrega o arquivo CSV com os dados expandidos e realiza cache dos dados para otimizar a performance, evitando recarregamentos desnecessários.

* Arquivo esperado: df_expandido.csv

3. Título Principal

```bash
st.title("Dashboard de Churn - Big Insights")

```
Exibe o título principal do dashboard.


4. Sidebar com Filtros Interativos
A sidebar contém filtros para refinar os dados visualizados:

```bash
st.sidebar.header("Filtros")

```
* Tempo de contrato (tenure): slider para definir o intervalo de meses do contrato.

* Valor mensal (valor_mensal): slider para definir o intervalo do valor mensal pago.

* SeniorCitizen: checkbox para filtrar apenas clientes idosos

```bash
tenure_min, tenure_max = int(df['tenure'].min()), int(df['tenure'].max())
tenure_range = st.sidebar.slider("Tempo de contrato (meses)", tenure_min, tenure_max, (tenure_min, tenure_max))

valor_mensal_min, valor_mensal_max = float(df['valor_mensal'].min()), float(df['valor_mensal'].max())
valor_mensal_range = st.sidebar.slider("Valor Mensal (R$)", valor_mensal_min, valor_mensal_max, (valor_mensal_min, valor_mensal_max))

senior = st.sidebar.checkbox("Mostrar somente clientes SeniorCitizen", value=False)

```

5. Aplicação dos Filtros
Os filtros são aplicados para criar um dataframe filtrado:

```bash
df_filtered = df[
    (df['tenure'] >= tenure_range[0]) &
    (df['tenure'] <= tenure_range[1]) &
    (df['valor_mensal'] >= valor_mensal_range[0]) &
    (df['valor_mensal'] <= valor_mensal_range[1])
]

if senior:
    df_filtered = df_filtered[df_filtered['SeniorCitizen'] == 1]

```
6. Exibição dos Dados Filtrados
Exibe o número de registros após os filtros e a tabela dos dados filtrados:

```bash
st.write(f"### Dados filtrados ({len(df_filtered)} registros)")
st.dataframe(df_filtered)

```
7. Estatísticas Rápidas
Métricas básicas exibidas em colunas:

* Clientes Totais: número total de clientes filtrados.

* Clientes que Cancelaram (Churn): total de clientes que tiveram churn = 'Yes'.

* Taxa de Churn (%): porcentagem de churn calculada no subset filtrado.

```bash
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Clientes Totais", len(df_filtered))

with col2:
    churn_count = df_filtered[df_filtered['Churn'] == 'Yes'].shape[0]
    st.metric("Clientes que Cancelaram (Churn)", churn_count)

with col3:
    churn_rate = 0 if len(df_filtered) == 0 else (churn_count / len(df_filtered)) * 100
    st.metric("Taxa de Churn (%)", f"{churn_rate:.2f}%")

 ```

### Como Executar

Na pasta do projeto, execute o comando:

```bash
streamlit run nome_do_arquivo.py

```
Substitua nome_do_arquivo.py pelo nome do script que contém o código do dashboard.

Próximos Passos
* Integração do modelo Random Forest para previsão de churn em tempo real com base nos filtros (ja incluído no dashboard do streamlit).

* Inclusão de gráficos interativos para análise visual de métricas e distribuição.

* Adição de mais filtros e segmentações para análises mais detalhadas.

# Documentação da Integração do Modelo Random Forest para Previsão de Churn

## Objetivo

Integrar ao dashboard Streamlit um modelo **Random Forest** previamente treinado e ajustado para realizar previsões de churn em tempo real, com base nos dados filtrados pelo usuário.

---

## Dependências Adicionais

Além das bibliotecas já usadas (pandas e streamlit), é necessário instalar:

```bash
pip install scikit-learn
pip install joblib

```
* scikit-learn: para manipulação do modelo Random Forest.

* joblib: para carregar o modelo salvo.

### Carregamento do Modelo
O modelo Random Forest treinado deve estar salvo em disco, por exemplo como random_forest_model.joblib.
```bash
import joblib

@st.cache_data
def load_model():
    model = joblib.load('random_forest_model.joblib')
    return model

model = load_model()

```

### Preparação dos Dados para Predição
* Selecionar as variáveis que o modelo utiliza (features).

* Garantir que os dados estejam no formato esperado pelo modelo (tratamento de variáveis categóricas, normalização se necessário).

* Exemplo simplificado:

```bash
features = ['tenure', 'valor_mensal', 'SeniorCitizen']  # lista das colunas usadas no modelo

X_filtered = df_filtered[features]

```


### Realizando a Predição
Com o modelo carregado e os dados preparados, aplicar a predição:

```bash
df_filtered['Churn_Prediction'] = model.predict(X_filtered)
df_filtered['Probabilidade_Churn'] = model.predict_proba(X_filtered)[:, 1]

```

* Churn_Prediction: previsão binária (Yes/No ou 1/0).

* Probabilidade_Churn: probabilidade da classe "churn".

### Exibição dos Resultados no Dashboard
Adicionar uma seção no dashboard para exibir os resultados da predição:

```bash
st.write("### Previsão de Churn nos Clientes Filtrados")

st.dataframe(df_filtered[['cliente_id', 'Churn_Prediction', 'Probabilidade_Churn']])

```

### Métricas com Base na Predição
Também é possível apresentar métricas agregadas sobre a predição, como:

```bash
col1, col2 = st.columns(2)

with col1:
    st.metric("Clientes com Predição de Churn", (df_filtered['Churn_Prediction'] == 1).sum())

with col2:
    taxa_predicao = 0 if len(df_filtered) == 0 else ((df_filtered['Churn_Prediction'] == 1).sum() / len(df_filtered)) * 100
    st.metric("Taxa de Churn Prevista (%)", f"{taxa_predicao:.2f}%")

```
### Como Usar
Prepare seu dataset df_expandido.csv com os dados já tratados e com as features corretas.

Salve seu modelo treinado com joblib.dump(model, 'random_forest_model.joblib').

Execute o Streamlit com o script atualizado.

# 📊 Interpretação da Saída do Dashboard de Churn - Big Insights

Este painel interativo permite explorar e analisar o comportamento de cancelamento (churn) de clientes da Telecom X, com base em variáveis-chave do relacionamento com a empresa.

---

## 🎯 Métricas Principais

| Indicador                         | Descrição                                                                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| **Clientes Totais**              | Quantidade de clientes exibidos após aplicar os filtros definidos na barra lateral. Representa o total da amostra visualizada.                 |
| **Clientes que Cancelaram (Churn)** | Quantidade de clientes da amostra filtrada que cancelaram o serviço (rótulo `Churn = Yes`). Indica a perda de clientes nesse grupo.           |
| **Taxa de Churn (%)**           | Proporção percentual de clientes que cancelaram, calculada como: `(Clientes com Churn / Clientes Totais) × 100`. Alta taxa indica alerta.     |

> ⚠️ **Interpretação**: Uma taxa de churn elevada em um segmento indica problemas potenciais com retenção, podendo exigir ações corretivas específicas para esse grupo.

---

## 🛠️ Painel Lateral de Filtros

| Filtro                            | Descrição                                                                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| **Tempo de contrato (meses)**    | Intervalo de tempo (tenure) que representa há quanto tempo o cliente mantém o contrato com a empresa. Valores baixos indicam novos clientes.   |
| **Valor Mensal (R$)**            | Faixa de valor da fatura mensal do cliente. Pode indicar diferentes perfis de planos (básico, intermediário, premium).                         |
| **Mostrar somente clientes SeniorCitizen** | Filtra para mostrar apenas clientes classificados como idosos (`SeniorCitizen = 1`). Útil para análises demográficas.                         |

> 💡 **Dica de uso**: Combine diferentes filtros para analisar segmentos específicos, como clientes idosos com baixo tempo de contrato e alto valor mensal, que podem ter maior propensão ao churn.

---

## 📌 Exemplo de Análise

Imagine o seguinte cenário após aplicar filtros:
- Tempo de contrato entre 1 e 6 meses
- Valor mensal entre R$ 60 e R$ 150
- Apenas clientes idosos (SeniorCitzens)

**Resultado observado:**
- Clientes Totais: 155  
- Clientes que Cancelaram: 117 
- Taxa de Churn: 75,48 %

🔎 **Interpretação**: Neste segmento, 7,5 a cada 10 clientes cancelaram, sugerindo que clientes idosos, com pouco tempo de contrato e alto custo mensal, estão mais propensos ao churn. Estratégias de retenção específicas devem ser analisadas para esse grupo.

---

## ✅ Conclusão

O painel fornece uma maneira poderosa e visual de analisar o churn por segmento, facilitando a tomada de decisões estratégicas de retenção de clientes com base em dados reais.

---

## 🗺️  Fluxograma simplificado do processo
Aqui está o resumo visual do que o código faz, passo a passo:

📂 Carregar dados CSV

       ↓

🎯 Definir variável-alvo: "Churn"
      
       ↓

🔢 Converter "Yes"/"No" para 1/0
       
       ↓

🧹 Limpar dados faltantes
       
       ↓

📊 Separar variáveis: Numéricas e Categóricas
       
       ↓

🔧 Pré-processar dados (OneHotEncoder)
       
       ↓

🧪 Dividir em treino (70%) e teste (30%)
       
       ↓

⚖️ Aplicar SMOTE (balancear classes)
       
       ↓

🌲 Treinar modelo Random Forest com GridSearchCV
       
       ↓

✅ Avaliar modelo (acurácia, relatório, confusão)
       
       ↓
💾 Salvar modelo e pré-processador
       
       ↓
⬇️ Fazer download (se estiver no Colab)


### 👨‍💻 Sobre o Desenvolvedor

**Luiz André de Souza**  
📍 GitHub: [@brodyandre](https://github.com/brodyandre)

> *Ralei bastante pra conseguir chegar até aqui!!!*  
> Tentando apresentar um projeto completo baseado em dados, com início, meio e fim, com a expectativa de criar um projeto de alto nível, que pudesse contribuir com a comunidade científica.  
> Claro que ele não é perfeito! Mas acredito que juntos podemos ir mais longe.  
> **Venha, contribua. Vamos tornar esse projeto uma referência!**

---

### 🚀 Como Contribuir

[![GitHub issues](https://img.shields.io/github/issues/brodyandre/churn_telecomx)](https://github.com/brodyandre/churn_telecomx/issues)
[![GitHub forks](https://img.shields.io/github/forks/brodyandre/churn_telecomx)](https://github.com/brodyandre/churn_telecomx/network)
[![GitHub stars](https://img.shields.io/github/stars/brodyandre/churn_telecomx)](https://github.com/brodyandre/churn_telecomx/stargazers)
[![GitHub license](https://img.shields.io/github/license/brodyandre/churn_telecomx)](https://github.com/brodyandre/churn_telecomx/blob/main/LICENSE)

💡 Tem uma ideia para melhorar este projeto?  
1. Faça um fork  
2. Crie uma nova branch (`git checkout -b melhoria-minha`)  
3. Commit suas mudanças (`git commit -m 'Sugestão de melhoria'`)  
4. Dê um push (`git push origin melhoria-minha`)  
5. Abra um **Pull Request**

📩 Ou abra uma [issue aqui](https://github.com/brodyandre/churn_telecomx/issues) para sugerir discussões, melhorias ou relatar bugs.

---

