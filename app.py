import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Churn - Big Insights", layout="wide")

# Carregar dados e gerar variáveis dummy
@st.cache_data
def load_data():
    df = pd.read_csv('df_expandido.csv')

    # Criar variáveis dummy para colunas categóricas relevantes
    dummies = pd.get_dummies(df[['Contract', 'TechSupport', 'PaymentMethod', 'OnlineBackup']], drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    return df

# Função para treinar o modelo com GridSearch
@st.cache_data(show_spinner=True)
def train_model(df):
    # Mapear target binário (simples): Churn Yes=1, No=0
    df = df.copy()
    df['Churn_binary'] = df['Churn'].map({'Yes':1, 'No':0})

    # Features importantes (numéricas + dummies)
    feature_cols = [
        'tenure', 'valor_mensal',
        'Contract_Month-to-month', 'TechSupport_No',
        'PaymentMethod_Electronic check', 'OnlineBackup_No',
        'SeniorCitizen'
    ]
    X = df[feature_cols]
    y = df['Churn_binary']

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Pipeline com scaler + RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Parametros para GridSearch
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [5, 10, None],
        'clf__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    return grid_search, X_test, y_test

df = load_data()

st.title("Dashboard de Churn - Big Insights")

# --- Sidebar para filtros ---
st.sidebar.header("Filtros")

# Filtro tenure (tempo de contrato)
tenure_min = int(df['tenure'].min())
tenure_max = int(df['tenure'].max())
tenure_range = st.sidebar.slider(
    "Tempo de contrato (meses)",
    min_value=tenure_min,
    max_value=tenure_max,
    value=(tenure_min, tenure_max)
)

# Filtro valor mensal
valor_mensal_min = round(float(df['valor_mensal'].min()), 2)

# Valor máximo fixo em 300,00
valor_mensal_max = 300.00

valor_mensal_range = st.sidebar.slider(
    "Valor Mensal (R$)",
    min_value=valor_mensal_min,
    max_value=valor_mensal_max,
    value=(valor_mensal_min, valor_mensal_max)
)

# Filtro SeniorCitizen (checkbox)
senior_only = st.sidebar.checkbox("Mostrar somente clientes SeniorCitizen", value=False)

# Novos filtros checkbox para variáveis dummy
contract_monthly = st.sidebar.checkbox("Contract Month-to-month", value=False)
techsupport_no = st.sidebar.checkbox("TechSupport No", value=False)
paymentmethod_echeck = st.sidebar.checkbox("PaymentMethod Electronic check", value=False)
onlinebackup_no = st.sidebar.checkbox("OnlineBackup No", value=False)

# Aplicar filtros no dataframe
df_filtered = df[
    (df['tenure'] >= tenure_range[0]) &
    (df['tenure'] <= tenure_range[1]) &
    (df['valor_mensal'] >= valor_mensal_range[0]) &
    (df['valor_mensal'] <= valor_mensal_range[1])
]

if senior_only:
    df_filtered = df_filtered[df_filtered['SeniorCitizen'] == 1]

if contract_monthly:
    df_filtered = df_filtered[df_filtered['Contract_Month-to-month'] == 1]

if techsupport_no:
    df_filtered = df_filtered[df_filtered['TechSupport_No'] == 1]

if paymentmethod_echeck:
    df_filtered = df_filtered[df_filtered['PaymentMethod_Electronic check'] == 1]

if onlinebackup_no:
    df_filtered = df_filtered[df_filtered['OnlineBackup_No'] == 1]

# Colunas importantes para visualização
colunas_importantes = [
    'id_cliente', 'Churn', 'tenure', 'Contract_Month-to-month',
    'TechSupport_No', 'PaymentMethod_Electronic check', 'OnlineBackup_No'
]

df_exibicao = df_filtered[colunas_importantes]

st.write(f"### Dados filtrados com variáveis importantes ({len(df_exibicao)} registros)")
st.dataframe(df_exibicao)

# Estatísticas básicas
st.write("### Estatísticas rápidas")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Clientes Totais", len(df_exibicao))

with col2:
    churn_count = df_exibicao[df_exibicao['Churn'] == 'Yes'].shape[0]
    st.metric("Clientes que Cancelaram (Churn)", churn_count)

with col3:
    churn_rate = 0 if len(df_exibicao) == 0 else (churn_count / len(df_exibicao)) * 100
    st.metric("Taxa de Churn (%)", f"{churn_rate:.2f}%")

st.write("---")

# Treinar modelo
st.write("### Treinamento e avaliação do modelo")
with st.spinner('Treinando o modelo... isso pode levar alguns segundos.'):

    grid_search, X_test, y_test = train_model(df)

    st.write(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

    # Probabilidades para a classe positiva
    y_prob = grid_search.predict_proba(X_test)[:, 1]

    # Avaliar e exibir métricas para diferentes thresholds
    def avaliar_threshold(threshold, y_prob, y_true):
        y_pred = (y_prob >= threshold).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True)
        conf_mat = confusion_matrix(y_true, y_pred)
        acc = (y_pred == y_true).mean()

        st.write(f"**Threshold = {threshold:.2f}**")
        st.write(f"Acurácia: {acc:.4f}")
        st.write("Relatório de Classificação:")
        st.json(report)
        st.write("Matriz de Confusão:")
        st.write(conf_mat)

        return y_pred

    # Mostrar curvas ROC e Precision-Recall com matplotlib + st.pyplot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_roc = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
    ax[0].plot([0,1],[0,1],'k--')
    ax[0].set_xlabel('Falso Positivo')
    ax[0].set_ylabel('Verdadeiro Positivo')
    ax[0].set_title('Curva ROC')
    ax[0].legend()

    ax[1].plot(recall, precision, label=f'AP = {ap_score:.2f}')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Curva Precision-Recall')
    ax[1].legend()

    st.pyplot(fig)

    # Testar thresholds e mostrar avaliação
    thresholds_teste = [0.3, 0.4, 0.5, 0.6, 0.7]
    for th in thresholds_teste:
        avaliar_threshold(th, y_prob, y_test)

    # Threshold escolhido para demonstração
    threshold_escolhido = 0.4
    y_pred_ajustado = (y_prob >= threshold_escolhido).astype(int)

    st.write(f"### Avaliação final com threshold ajustado = {threshold_escolhido}")
    st.text(classification_report(y_test, y_pred_ajustado))
    st.write("Matriz de Confusão:")
    st.write(confusion_matrix(y_test, y_pred_ajustado))

