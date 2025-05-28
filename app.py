import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)

st.set_page_config(page_title="Dashboard Churn - Big Insights", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('df_expandido.csv')

    # Criar variáveis dummy para as colunas categóricas relevantes
    dummies = pd.get_dummies(df[['Contract', 'TechSupport', 'PaymentMethod', 'OnlineBackup']], drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    return df

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
valor_mensal_max = round(float(df['valor_mensal'].max()), 2)
valor_mensal_range = st.sidebar.slider(
    "Valor Mensal (R$)",
    min_value=valor_mensal_min,
    max_value=valor_mensal_max,
    value=(valor_mensal_min, valor_mensal_max)
)

# Filtro SeniorCitizen (checkbox)
senior = st.sidebar.checkbox("Mostrar somente clientes SeniorCitizen", value=False)

# Aplicar filtros no dataframe
df_filtered = df[
    (df['tenure'] >= tenure_range[0]) &
    (df['tenure'] <= tenure_range[1]) &
    (df['valor_mensal'] >= valor_mensal_range[0]) &
    (df['valor_mensal'] <= valor_mensal_range[1])
]

if senior:
    df_filtered = df_filtered[df_filtered['SeniorCitizen'] == 1]

# --- Seleção apenas das colunas importantes ---
colunas_importantes = [
    'id_cliente', 'Churn', 'tenure', 'Contract_Month-to-month',
    'TechSupport_No', 'PaymentMethod_Electronic check', 'OnlineBackup_No'
]

# Exibir as colunas mais relevantes após filtro
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

# --- Avaliação do modelo ---
# Supondo que você já tem o grid_search treinado e X_test_prep, y_test definidos
# 1. Obter probabilidades preditas para a classe positiva
y_prob = grid_search.predict_proba(X_test_prep)[:, 1]

# 2. Definir função para avaliar o modelo para diferentes thresholds
def avaliar_threshold(threshold, y_prob, y_true):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n=== Avaliação para threshold = {threshold:.2f} ===")
    print("Relatório de Classificação:")
    print(classification_report(y_true, y_pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
    acc = (y_pred == y_true).mean()
    print(f"Acurácia: {acc:.4f}")
    return y_pred

# 3. Plotar curva ROC e Precision-Recall
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
auc_roc = roc_auc_score(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC')
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()

plt.tight_layout()
plt.show()

# 4. Testar diferentes thresholds para achar um bom ponto de corte
thresholds_teste = [0.3, 0.4, 0.5, 0.6, 0.7]
for th in thresholds_teste:
    avaliar_threshold(th, y_prob, y_test)

# Você pode escolher o threshold que melhor balanceia recall e precision para seu objetivo
# Por exemplo, suponha que 0.4 é um bom valor:
threshold_escolhido = 0.4
y_pred_ajustado = (y_prob >= threshold_escolhido).astype(int)

print(f"\n=== Avaliação final com threshold ajustado = {threshold_escolhido} ===")
print(classification_report(y_test, y_pred_ajustado))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_ajustado))

