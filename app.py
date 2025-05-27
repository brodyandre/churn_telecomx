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

# Filtro tenure
tenure_min, tenure_max = int(df['tenure'].min()), int(df['tenure'].max())
tenure_range = st.sidebar.slider("Tempo de contrato (meses)", tenure_min, tenure_max, (tenure_min, tenure_max))

# Filtro valor mensal - faixa ampliada
valor_mensal_min, valor_mensal_max = float(df['valor_mensal'].min()), float(df['valor_mensal'].max())
faixa_min = max(0, valor_mensal_min - 20)
faixa_max = valor_mensal_max + 20
valor_mensal_range = st.sidebar.slider("Valor Mensal (R$)", faixa_min, faixa_max, (valor_mensal_min, valor_mensal_max))

# Filtro SeniorCitizen
senior = st.sidebar.checkbox("Mostrar somente clientes SeniorCitizen", value=False)

# Novos filtros com base nas variáveis mais importantes
st.sidebar.markdown("### Variáveis mais influentes")

# Filtro Contract
contracts = df['Contract'].unique().tolist()
contract_selected = st.sidebar.multiselect("Tipo de Contrato", contracts, default=contracts)

# Filtro TechSupport
tech_support_options = df['TechSupport'].unique().tolist()
tech_selected = st.sidebar.multiselect("Suporte Técnico", tech_support_options, default=tech_support_options)

# Filtro PaymentMethod
payment_methods = df['PaymentMethod'].unique().tolist()
payment_selected = st.sidebar.multiselect("Método de Pagamento", payment_methods, default=payment_methods)

# Filtro OnlineBackup
backup_options = df['OnlineBackup'].unique().tolist()
backup_selected = st.sidebar.multiselect("Backup Online", backup_options, default=backup_options)

# --- Aplicar Filtros ---
df_filtered = df[
    (df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1]) &
    (df['valor_mensal'] >= valor_mensal_range[0]) & (df['valor_mensal'] <= valor_mensal_range[1]) &
    (df['Contract'].isin(contract_selected)) &
    (df['TechSupport'].isin(tech_selected)) &
    (df['PaymentMethod'].isin(payment_selected)) &
    (df['OnlineBackup'].isin(backup_selected))
]

if senior:
    df_filtered = df_filtered[df_filtered['SeniorCitizen'] == 1]

# --- Dashboard ---
st.write(f"### Dados filtrados ({len(df_filtered)} registros)")
st.dataframe(df_filtered)

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

