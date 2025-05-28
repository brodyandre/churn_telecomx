import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Churn - Big Insights", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('df_expandido.csv')
    return df

df = load_data()

st.title("Dashboard de Churn - Big Insights")

# Sidebar - filtros
st.sidebar.header("Filtros")

# Filtro tenure (numérico)
tenure_min, tenure_max = int(df['tenure'].min()), int(df['tenure'].max())
tenure_range = st.sidebar.slider("Tempo de contrato (meses)", tenure_min, tenure_max, (tenure_min, tenure_max))

# Filtro valor mensal (numérico) com faixa ampliada
valor_mensal_min, valor_mensal_max = float(df['valor_mensal'].min()), float(df['valor_mensal'].max())
faixa_min = max(0, valor_mensal_min - 20)
faixa_max = valor_mensal_max + 20
default_min = max(faixa_min, valor_mensal_min)
default_max = min(faixa_max, valor_mensal_max)
valor_mensal_range = st.sidebar.slider(
    "Valor Mensal (R$)", min_value=faixa_min, max_value=faixa_max, value=(default_min, default_max)
)

# Filtro SeniorCitizen (binário)
senior = st.sidebar.checkbox("Mostrar somente clientes SeniorCitizen", value=False)

# Filtro variáveis categóricas/binárias importantes
st.sidebar.subheader("Filtros adicionais")

# Contract_Month-to-month (sim ou não)
contract_month_to_month = st.sidebar.selectbox(
    "Contrato Month-to-month",
    options=["Todos", "Sim", "Não"]
)

# TechSupport_No (sim ou não)
techsupport_no = st.sidebar.selectbox(
    "Sem Suporte Técnico",
    options=["Todos", "Sim", "Não"]
)

# PaymentMethod_Electronic check (sim ou não)
payment_electronic = st.sidebar.selectbox(
    "Pagamento por Cheque Eletrônico",
    options=["Todos", "Sim", "Não"]
)

# OnlineBackup_No (sim ou não)
online_backup_no = st.sidebar.selectbox(
    "Sem Backup Online",
    options=["Todos", "Sim", "Não"]
)

# Aplicar filtros
df_filtered = df[
    (df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1]) &
    (df['valor_mensal'] >= valor_mensal_range[0]) & (df['valor_mensal'] <= valor_mensal_range[1])
]

if senior:
    df_filtered = df_filtered[df_filtered['SeniorCitizen'] == 1]

# Contract_Month-to-month filtro
if contract_month_to_month == "Sim":
    df_filtered = df_filtered[df_filtered['Contract_Month-to-month'] == 1]
elif contract_month_to_month == "Não":
    df_filtered = df_filtered[df_filtered['Contract_Month-to-month'] == 0]

# TechSupport_No filtro
if techsupport_no == "Sim":
    df_filtered = df_filtered[df_filtered['TechSupport_No'] == 1]
elif techsupport_no == "Não":
    df_filtered = df_filtered[df_filtered['TechSupport_No'] == 0]

# PaymentMethod_Electronic check filtro
if payment_electronic == "Sim":
    df_filtered = df_filtered[df_filtered['PaymentMethod_Electronic check'] == 1]
elif payment_electronic == "Não":
    df_filtered = df_filtered[df_filtered['PaymentMethod_Electronic check'] == 0]

# OnlineBackup_No filtro
if online_backup_no == "Sim":
    df_filtered = df_filtered[df_filtered['OnlineBackup_No'] == 1]
elif online_backup_no == "Não":
    df_filtered = df_filtered[df_filtered['OnlineBackup_No'] == 0]

# Mostrar dados filtrados
st.write(f"### Dados filtrados ({len(df_filtered)} registros)")
st.dataframe(df_filtered)

# Estatísticas rápidas
st.write("### Estatísticas rápidas")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Clientes Totais", len(df_filtered))

with col2:
    churn_count = df_filtered[df_filtered['Churn'] == 'Yes'].shape[0]
    st.metric("Clientes que Cancelaram (Churn)", churn_count)

with col3:
    churn_rate = 0 if len(df_filtered) == 0 else (churn_count / len(df_filtered)) * 100
    st.metric("Taxa de Churn (%)", f"{churn_rate:.2f}%")

