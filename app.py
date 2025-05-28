import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Churn TELECOM X", layout="wide")

# Função para carregar os dados e gerar variáveis dummy
@st.cache_data
def load_data():
    df = pd.read_csv("df_expandido.csv")

    # Criar variáveis dummy para colunas relevantes
    dummies = pd.get_dummies(df[['Contract', 'TechSupport', 'PaymentMethod', 'OnlineBackup']], drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    return df

# Carrega os dados
df = load_data()

st.title("Análise de Churn - TELECOM X")

# Filtros laterais
st.sidebar.header("Filtros")

# Filtro 1: Contract_Month-to-month
contract_filter = st.sidebar.checkbox("Contrato: Mês a Mês", value=True)

# Filtro 2: TechSupport_No
techsupport_filter = st.sidebar.checkbox("Sem Suporte Técnico", value=True)

# Filtro 3: PaymentMethod_Electronic check
payment_filter = st.sidebar.checkbox("Pagamento por Débito Automático", value=True)

# Filtro 4: OnlineBackup_No
backup_filter = st.sidebar.checkbox("Sem Backup Online", value=True)

# Filtro 5: Faixa de valor mensal
faixa_min = float(df['valor_mensal'].min())
faixa_max = float(df['valor_mensal'].max())
default_min = float(df['valor_mensal'].quantile(0.25))
default_max = float(df['valor_mensal'].quantile(0.75))

valor_mensal_range = st.sidebar.slider(
    "Valor Mensal (R$)", 
    min_value=faixa_min, 
    max_value=faixa_max, 
    value=(default_min, default_max)
)

# Filtro 6: Faixa de tempo de permanência
tenure_min = int(df['tenure'].min())
tenure_max = int(df['tenure'].max())
default_tenure_min = int(df['tenure'].quantile(0.25))
default_tenure_max = int(df['tenure'].quantile(0.75))

tenure_range = st.sidebar.slider(
    "Tempo de Permanência (meses)", 
    min_value=tenure_min, 
    max_value=tenure_max, 
    value=(default_tenure_min, default_tenure_max)
)

# Aplicar filtros ao DataFrame
filtered_df = df[
    (df['valor_mensal'] >= valor_mensal_range[0]) &
    (df['valor_mensal'] <= valor_mensal_range[1]) &
    (df['tenure'] >= tenure_range[0]) &
    (df['tenure'] <= tenure_range[1])
]

if contract_filter:
    filtered_df = filtered_df[filtered_df['Contract_Month-to-month'] == 1]

if techsupport_filter:
    filtered_df = filtered_df[filtered_df['TechSupport_No'] == 1]

if payment_filter:
    filtered_df = filtered_df[filtered_df['PaymentMethod_Electronic check'] == 1]

if backup_filter:
    filtered_df = filtered_df[filtered_df['OnlineBackup_No'] == 1]

# Exibir resultados filtrados
st.subheader("Clientes Filtrados")
st.write(f"Total de clientes após filtros: {filtered_df.shape[0]}")
st.dataframe(filtered_df[['id_cliente', 'Churn', 'valor_mensal', 'tenure',
                          'Contract', 'TechSupport', 'PaymentMethod', 'OnlineBackup']])
