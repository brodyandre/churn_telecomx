import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Churn - Big Insights", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('df_expandido.csv')
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
