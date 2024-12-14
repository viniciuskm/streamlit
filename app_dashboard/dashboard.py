import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys
import os
# import locale
import gdown

# URL pública do Google Drive
url = "https://drive.google.com/uc?id=1m1SnhX6yQ6PNXQwZ-MaR9U4H-1ix-Nf2"
output = "dataprep_model_v2.pkl"

# Baixa o arquivo, se ainda não estiver no diretório local
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)


# URL pública do Google Drive
url = "https://drive.google.com/uc?id=1RTvheYNiIgOxUbHgCxChWYBEX7XlSpjc"
output = "dataprep_full_v2.pkl"

# Baixa o arquivo, se ainda não estiver no diretório local
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)



# Configurar para formato monetário brasileiro
# locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# Adicionar o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.indice_liquidez_v2 import calcular_indice_liquidez_otimizado_v2
from src.utils.prep_model_variables_v3 import main_prep_variables

# Função para validar e processar a entrada de coordenadas
def parse_coordinates(coord_input):
    try:
        lat, lon = map(float, coord_input.split(","))
        return lat, lon
    except ValueError:
        st.error("Coordenadas inválidas! Insira no formato: latitude, longitude (ex: -23.61338393127851, -46.68320575925552)")
        return None, None

# Configuração inicial do Streamlit
st.set_page_config(page_title="Consulta de Empreendimentos", layout="wide")
st.title("Consulta de Empreendimentos por Localização")

# Carregar os dados
# pkl_path = "models/01_arquivofunctionprep/data/dataprep_full_v2.pkl"
pkl_path = "dataprep_full_v2.pkl"

try:
    df = pd.read_pickle(pkl_path)
    # df['data'] = df['data de transação'].dt.strftime("%d/%m/%Y")
except FileNotFoundError:
    st.error(f"O arquivo {pkl_path} não foi encontrado. Certifique-se de que o caminho está correto.")
    st.stop()

# Verificar se o DataFrame contém as colunas necessárias
required_columns = ['data de transação', 'nome do logradouro','número','complemento','referência','acc (iptu)',
                    'valor de transação (declarado pelo contribuinte)','vlr_trans_comp_ipca','preco_m2_trans_ipca',
                    'área construída (m2)', 'fração ideal', 'área do terreno (m2)','metro_1km','shopping_dentro_1km']

# Inicializar st.session_state para manter os resultados
if "gdf_proximos" not in st.session_state:
    st.session_state["gdf_proximos"] = None

if "gdf_proximos_msmnumero" not in st.session_state:
    st.session_state["gdf_proximos_msmnumero"] = None

if "previsao_precos_clicado" not in st.session_state:
    st.session_state["previsao_precos_clicado"] = False

if "indice_liquidez" not in st.session_state:
    st.session_state["indice_liquidez"] = None
    
    
    
# Função para processar os dados e chamar a função de preparação
def preparar_dados_para_previsao(latitude, longitude, ano_construcao, fracao_ideal, area_construida, area_terreno):
    try:
        # Criar um DataFrame com os dados fornecidos pelo usuário
        df_usuario = pd.DataFrame([{
            'latitude': latitude,
            'longitude': longitude,
            'ano_construcao': ano_construcao,
            'fracao_ideal': fracao_ideal,
            'area_construida': area_construida,
            'area_terreno': area_terreno
        }])
        df_usuario['data_transacao'] = '2024-09-01'

        # Chamar a função main_prep_variables
        return main_prep_variables(
            df_imoveis=df_usuario,
            col_lat='latitude',
            col_lon='longitude',
            col_ano_construcao='ano_construcao',
            col_fracao_ideal='fracao_ideal',
            col_area_construida='area_construida',
            col_area_terreno='area_terreno',
            data_transacao='2024-09-01'
        )
    except Exception as e:
        st.error(f"Erro ao preparar os dados para previsão: {e}")
        return None


# Formulário para "Empreendimentos Próximos"
st.sidebar.header("Empreendimentos Próximos", divider='red')
with st.sidebar.form(key="form_filtros"):
    coord_input = st.text_input("Insira as coordenadas (latitude, longitude)", value="-23.61338393127851, -46.68320575925552")
    numero_emp = st.number_input("Número do Empreendimento", value=539, step=1)
    raio = st.number_input("Raio em metros", value=200, step=100)
    aplicar_filtros = st.form_submit_button("Aplicar Filtros")

# Variáveis processadas do formulário "Empreendimentos Próximos"
latitude, longitude = parse_coordinates(coord_input)

if aplicar_filtros and latitude is not None and longitude is not None:
    try:
        ponto_usuario = Point(float(longitude), float(latitude))
        ponto_usuario_gdf = gpd.GeoDataFrame(
            {'geometry': [ponto_usuario]}, 
            crs="EPSG:4326"
        ).to_crs(epsg=32723)
        buffer_dist = ponto_usuario_gdf.buffer(raio)
        gdf_proximos = df[df.geometry.intersects(buffer_dist.unary_union)]
        st.session_state["gdf_proximos"] = gdf_proximos
        ponto_usuario_gdf[['indice_liquidez', 'liquidez_cluster']] = calcular_indice_liquidez_otimizado_v2(ponto_usuario_gdf, gdf_proximos)
        ponto_usuario_gdf[['latitude','longitude']] = [latitude, longitude]

        st.session_state["indice_liquidez"] = ponto_usuario_gdf
        gdf_proximos_msmnumero = gdf_proximos[gdf_proximos['número'] == numero_emp]
        st.session_state["gdf_proximos_msmnumero"] = gdf_proximos_msmnumero
    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")

# Formulário para "Previsão de Preço"
st.sidebar.header("Características para Previsão de Preço", divider='red')
with st.sidebar.form(key="form_previsao"):
    ano_construcao = st.number_input("Ano de Construção", min_value=1800, max_value=2024, value=2000, step=1)
    fracao_ideal = st.number_input("Fração Ideal (ex: 0.001)", min_value=0.0, format="%.5f")
    area_construida = st.number_input("Área Construída (m²)", min_value=0.0, step=0.1)
    area_terreno = st.number_input("Área do Terreno (m²)", min_value=0.0, step=0.1)
    prever_preco = st.form_submit_button("Prever Preço")

if prever_preco:
    st.session_state["previsao_precos_clicado"] = True

tab1, tab2, tab3, tab4 = st.tabs(["Empreendimentos Próximos", "Índice de Liquidez", "Mesmo Número", "Previsão de Preço"])


with tab1:
    if st.session_state["gdf_proximos"] is not None:
        st.subheader(f"Empreendimentos Próximos (Raio de {raio}m)")
        st.write(st.session_state["gdf_proximos"][required_columns])
    else:
        st.write("Nenhum resultado disponível. Clique em 'Aplicar Filtros'.")

with tab2:
    if st.session_state["indice_liquidez"] is not None:
        st.subheader("Índice de Liquidez")
        st.write(st.session_state["indice_liquidez"].drop('geometry', axis=1))
    else:
        st.write("Nenhum resultado disponível. Clique em 'Aplicar Filtros'.")

with tab3:
    if st.session_state["gdf_proximos_msmnumero"] is not None:
        st.subheader(f"Empreendimentos Próximos e Mesmo Número (Raio de {raio}m)")
        st.write(st.session_state["gdf_proximos_msmnumero"][required_columns])
    else:
        st.write("Nenhum resultado disponível. Clique em 'Aplicar Filtros'.")


def formatar_moeda(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")



with tab4:
    if st.session_state["previsao_precos_clicado"]:
        # Passar variáveis do formulário para a função de previsão
        st.write('Prevendo preço...')
        df_predict = preparar_dados_para_previsao(latitude, longitude, ano_construcao, fracao_ideal, area_construida, area_terreno)
        if df_predict is not None:
            # valor_predito = locale.currency(df_predict['vlr_predito'].values[0], grouping=True)
            valor_predito = df_predict['vlr_predito'].values[0]
            st.subheader(f"Valor Predito: {formatar_moeda(valor_predito)}")
            st.write("Dados utilizados no modelo de previsão:")
            st.write(df_predict)
        else:
            st.error("Erro ao preparar os dados para previsão.")
    else:
        st.write("Clique em 'Prever Preço' na barra lateral para gerar a previsão.")

