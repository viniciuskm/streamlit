import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
import datetime
import os
import joblib
from bcb import sgs
# from sklearn.model_selection import train_test_split
# from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from src.paths.config_paths import p_model_xgb_fullvariables, p_model_equipamentos_saopaulo, p_model_dataprep, p_model_equipamentos_fipezap
from tqdm import tqdm


# =============================================================================
# Paths
# =============================================================================

# Definindo o caminho para os dados brutos
# path_data_raw = '../previsao_precos_imoveis_model/data/raw/01_equipamentos_saopaulo'
# path_dados_dataprep = '../previsao_precos_imoveis_model/data/processed/prep_model/dataprep_model_v2.pkl'
# modelo_path = '../previsao_precos_imoveis_model/models/version_1/best_xgb_model.pkl'

path_data_raw = p_model_equipamentos_saopaulo
# path_dados_dataprep = p_model_dataprep
path_dados_dataprep = 'dataprep_model_v2.pkl'
modelo_path = p_model_xgb_fullvariables
# file_path = '01_data_raw/indice_fipezap/fipezap-serieshistoricas.xlsx'
file_path = p_model_equipamentos_fipezap

# =============================================================================
# 
# =============================================================================

# Função para calcular as variáveis necessárias para o modelo
def calcular_variaveis(lat, 
                       lon, 
                       ano_construcao,
                       area_terreno,
                       fracao_ideal,
                       area_construida,
                       gdf_parque, 
                       gdf_metro, 
                       gdf_escolas, 
                       gdf_hospital, 
                       gdf_shopping, 
                       gdf_dados_treino):
    # Criar GeoDataFrame com o imóvel fornecido
    geometry = Point(lon, lat)
    gdf_imovel = gpd.GeoDataFrame({'lat_goog': [lat], 'lon_goog': [lon], 'geometry': [geometry]}, crs="EPSG:4326")
    gdf_imovel = gdf_imovel.to_crs(epsg=32723)  # Converter para o sistema de coordenadas usado
    
    
    ano_atual = datetime.datetime.today().year
    idade = ano_atual - ano_construcao

    
    def ajustar_fracao_ideal(fracao_ideal, gdf_dados_treino):
        if pd.isnull(fracao_ideal):
            return gdf_dados_treino['fração ideal'].mean()
        return fracao_ideal



    # Ajustar a fração ideal usando a função interna
    fracao_ideal = ajustar_fracao_ideal(fracao_ideal, gdf_dados_treino)

    
    # Calcular variáveis baseadas em localização
    def contar_escolas(gdf_escolas, empreendimento_geom, raio, privada):
        if privada == 1:
            escolas_dentro_raio = gdf_escolas[(gdf_escolas.geometry.distance(empreendimento_geom) <= raio) & (gdf_escolas['eq_classe'] == "REDE_PRIVADA")]
        else:
            escolas_dentro_raio = gdf_escolas[(gdf_escolas.geometry.distance(empreendimento_geom) <= raio) & (gdf_escolas['eq_classe'] != "REDE_PRIVADA")]
        return len(escolas_dentro_raio)

    def existe_metro_no_raio(gdf_metro, empreendimento_geom, raio):
        distancias = gdf_metro.loc[gdf_metro.emt_empres=='METRO'].geometry.distance(empreendimento_geom)
        return any(distancias <= raio)

    def existe_trem_no_raio(gdf_metro, empreendimento_geom, raio):
        distancias = gdf_metro.loc[gdf_metro.emt_empres!='METRO'].geometry.distance(empreendimento_geom)
        return any(distancias <= raio)
    
    def contar_hospitais(gdf_hospital, empreendimento_geom, raio):
        hospitais_dentro_raio = gdf_hospital[(gdf_hospital.geometry.distance(empreendimento_geom) <= raio) &
                                             (gdf_hospital['eq_classe'] == "HOSPITAL")]
        return len(hospitais_dentro_raio)

    def contar_UBS(gdf_hospital, empreendimento_geom, raio):
        ubs_dentro_raio = gdf_hospital[(gdf_hospital.geometry.distance(empreendimento_geom) <= raio) &
                                       (gdf_hospital['eq_classe'] == "UBS_POSTO_DE_SAUDE_CENTRO_DE_SAUDE")]
        return len(ubs_dentro_raio)

    # Calcular variáveis
    escolas_priv_1km = contar_escolas(gdf_escolas, gdf_imovel.geometry.iloc[0], 1000, privada=1)
    escolas_priv_2km = contar_escolas(gdf_escolas, gdf_imovel.geometry.iloc[0], 2000, privada=1)
    escolas_pub_1km = contar_escolas(gdf_escolas, gdf_imovel.geometry.iloc[0], 1000, privada=0)
    escolas_pub_2km = contar_escolas(gdf_escolas, gdf_imovel.geometry.iloc[0], 2000, privada=0)
    
    metro_1km = existe_metro_no_raio(gdf_metro, gdf_imovel.geometry.iloc[0], 1000)
    metro_2km = existe_metro_no_raio(gdf_metro, gdf_imovel.geometry.iloc[0], 2000)
    metro_3km = existe_metro_no_raio(gdf_metro, gdf_imovel.geometry.iloc[0], 3000)
    
    trem_1km = existe_trem_no_raio(gdf_metro, gdf_imovel.geometry.iloc[0], 1000)
    trem_2km = existe_trem_no_raio(gdf_metro, gdf_imovel.geometry.iloc[0], 2000)
    trem_3km = existe_trem_no_raio(gdf_metro, gdf_imovel.geometry.iloc[0], 3000)
    
    hospitais_1km = contar_hospitais(gdf_hospital, gdf_imovel.geometry.iloc[0], 1000)
    hospitais_2km = contar_hospitais(gdf_hospital, gdf_imovel.geometry.iloc[0], 2000)
    hospitais_3km = contar_hospitais(gdf_hospital, gdf_imovel.geometry.iloc[0], 3000)
    
    UBS_1km = contar_UBS(gdf_hospital, gdf_imovel.geometry.iloc[0], 1000)
    UBS_2km = contar_UBS(gdf_hospital, gdf_imovel.geometry.iloc[0], 2000)
    UBS_3km = contar_UBS(gdf_hospital, gdf_imovel.geometry.iloc[0], 3000)
    
    # Calcular distância para o shopping mais próximo
    distancia_shopping_mais_proximo = gdf_shopping.distance(gdf_imovel.geometry.iloc[0]).min()

    # Cálculo das médias e medianas dos imóveis mais próximos usando a base de dados de treinamento
    coords = gdf_dados_treino.geometry.apply(lambda point: (point.x, point.y)).tolist()
    nbrs = NearestNeighbors(n_neighbors=31, algorithm='ball_tree').fit(coords)  # Considera até 30 vizinhos mais próximos
    distances, indices = nbrs.kneighbors([(gdf_imovel.geometry.iloc[0].x, gdf_imovel.geometry.iloc[0].y)])

    def calcular_media_mediana(indices, gdf, num_vizinhos):
        neighbor_prices = gdf.iloc[indices[0][1:num_vizinhos+1]]['preco_m2_trans_ipca']
        # neighbor_prices = gdf.loc[indices[0][1:num_vizinhos+1], 'preco_m2_trans_ipca']
        media = neighbor_prices.mean()
        mediana = neighbor_prices.median()
        return media, mediana

    media_preco_10_mais_proximos, mediana_preco_10_mais_proximos = calcular_media_mediana(indices, gdf_dados_treino, 10)
    media_preco_20_mais_proximos, mediana_preco_20_mais_proximos = calcular_media_mediana(indices, gdf_dados_treino, 20)
    media_preco_30_mais_proximos, mediana_preco_30_mais_proximos = calcular_media_mediana(indices, gdf_dados_treino, 30)

    def calcular_area_parques_urbanos(gdf_parque, geometria, raio):
        parques_urbanos = gdf_parque[gdf_parque['cpuc_categ'] == 'Parque Urbano']
        parques_urbanos['cpuc_metro'] = pd.to_numeric(parques_urbanos['cpuc_metro'], errors='coerce').fillna(0)
        buffer = geometria.buffer(raio)
        parques_dentro_raio = parques_urbanos[parques_urbanos.geometry.intersects(buffer)]
        soma_area = parques_dentro_raio['cpuc_metro'].sum()
        return soma_area
    
    soma_area_parques_urbanos_2km = calcular_area_parques_urbanos(gdf_parque, gdf_imovel.geometry.iloc[0], 2000)

    
    # Criação do DataFrame final com as variáveis calculadas
    variaveis_calculadas = {
        'área do terreno (m2)': area_terreno,
        'área construída (m2)' : area_construida,
        'fração ideal': fracao_ideal,
        'idade': idade,
        'lat_goog': lat,
        'lon_goog': lon,
        'escolas_priv_1km': escolas_priv_1km,
        'escolas_priv_2km': escolas_priv_2km,
        'escolas_pub_1km': escolas_pub_1km,
        'escolas_pub_2km': escolas_pub_2km,
        'metro_1km': int(metro_1km),
        'metro_2km': int(metro_2km),
        'metro_3km': int(metro_3km),
        'trem_1km': int(trem_1km),
        'trem_2km': int(trem_2km),
        'trem_3km': int(trem_3km),
        'hospitais_1km': hospitais_1km,
        'hospitais_2km': hospitais_2km,
        'hospitais_3km': hospitais_3km,
        'UBS_1km': UBS_1km,
        'UBS_2km': UBS_2km,
        'UBS_3km': UBS_3km,
        'soma_area_parques_urbanos_2km':soma_area_parques_urbanos_2km,
        'distancia_shopping_mais_proximo': distancia_shopping_mais_proximo,
        'media_preco_10_mais_proximos': media_preco_10_mais_proximos,
        'mediana_preco_10_mais_proximos': mediana_preco_10_mais_proximos,
        'media_preco_20_mais_proximos': media_preco_20_mais_proximos,
        'mediana_preco_20_mais_proximos': mediana_preco_20_mais_proximos,
        'media_preco_30_mais_proximos': media_preco_30_mais_proximos,
        'mediana_preco_30_mais_proximos': mediana_preco_30_mais_proximos
    }

    return pd.DataFrame([variaveis_calculadas])



# =============================================================================
# Carregando dfs
# =============================================================================



# Função para carregar GeoDataFrames de referência
def carregar_gdfs_referencia():
    
    # Carregar dados de parques
    arquivos_shp_parques = [f for f in os.listdir(f'{path_data_raw}/parques') if f.endswith('.shp')]
    d_parques = {i: gpd.read_file(f'{path_data_raw}/parques/{i}') for i in arquivos_shp_parques}
    gdf_parque = pd.concat(d_parques).set_crs(epsg=32723)

    # Carregar dados de escolas
    arquivos_shp_escolas = [f for f in os.listdir(f'{path_data_raw}/escolas') if f.endswith('.shp')]
    d_escolas = {i: gpd.read_file(f'{path_data_raw}/escolas/{i}') for i in arquivos_shp_escolas}
    gdf_escolas = pd.concat(d_escolas).set_crs(epsg=32723)

    # Carregar dados de metrô e trem
    arquivos_shp_metro_trem = [f for f in os.listdir(f'{path_data_raw}/metro_trem') if f.endswith('.shp')]
    d_metro_trem = {i: gpd.read_file(f'{path_data_raw}/metro_trem/{i}').rename(columns = {'etr_nome':'emt_nome',
                                                                                          'etr_linha':'emt_linha',
                                                                                          'etr_empres':'emt_empres',
                                                                                          'etr_situac':'emt_situac'}) 
                    for i in arquivos_shp_metro_trem}
    gdf_metro = pd.concat(d_metro_trem).set_crs(epsg=32723)

    # Carregar dados de hospitais
    arquivos_shp_hospitais = [f for f in os.listdir(f'{path_data_raw}/hospitais') if f.endswith('.shp')]
    d_hospitais = {i: gpd.read_file(f'{path_data_raw}/hospitais/{i}') for i in arquivos_shp_hospitais}
    gdf_hospital = pd.concat(d_hospitais).set_crs(epsg=32723)

    # Carregar dados de shoppings
    arquivos_shp_shoppings = [f for f in os.listdir(f'{path_data_raw}/shoppings') if f.endswith('.shp')]
    d_shoppings = {i: gpd.read_file(f'{path_data_raw}/shoppings/{i}') for i in arquivos_shp_shoppings}
    gdf_shopping = pd.concat(d_shoppings).set_crs(epsg=32723)

    return gdf_parque, gdf_escolas, gdf_metro, gdf_hospital, gdf_shopping


# =============================================================================
# Calculo variaveis Macro
# =============================================================================

# Função para carregar e calcular as variáveis macroeconômicas
def calcular_variaveis_macroeconomicas(data_transacao):
    # Carregar série histórica do Banco Central do Brasil
    # IDs das séries: SELIC (432), IGPM (189), IPCA (433), INPC (188), Cambio (3695)
    serie_hist = sgs.get({
                            'selic': 432,
                           'igpm': 189, 
                           'ipca': 433,
                           'inpc': 188,
                           'cambio': 3695
                          },
                         start='2015-01-01').dropna()
    
    # Ajuste para valores percentuais dividindo por 100 onde necessário
    serie_hist[['selic', 'igpm', 'ipca', 'inpc', 'cambio']] /= 100
    # serie_hist[['selic', 'igpm', 'ipca', 'inpc']] /= 100

    # Ordenar séries históricas por data
    serie_hist = serie_hist.sort_index(ascending=True)
    
    # Calculando métricas solicitadas
    serie_hist['cambio_media_6m'] = serie_hist['cambio'].rolling(window=6).mean()
    serie_hist['selic_media_6m'] = serie_hist['selic'].rolling(window=6).mean()
    
    serie_hist['ipca_soma_12m'] = serie_hist['ipca'].rolling(window=12).sum()
    serie_hist['igpm_soma_12m'] = serie_hist['igpm'].rolling(window=12).sum()
    serie_hist['inpc_soma_12m'] = serie_hist['inpc'].rolling(window=12).sum()
    
    # Carregar o índice FipeZap do Excel
    df_fipezap = pd.read_excel(file_path, sheet_name='São Paulo', skiprows=3)
    df_fipezap = df_fipezap[['Data', 'Total']].rename(columns={'Data': 'Date', 'Total': 'indicefipezap'}).dropna()
    df_fipezap['Date'] = pd.to_datetime(df_fipezap['Date'])
    df_fipezap.set_index('Date', inplace=True)
    
    # Merge do índice FipeZap com a série histórica
    serie_hist = serie_hist.merge(df_fipezap, how='left', left_index=True, right_index=True)
    
    # Data de transação do imóvel (transformada em Periodo Mensal para comparação)
    data_transacao_period = pd.to_datetime(data_transacao).to_period('M')

    # Filtro para pegar os dados mais recentes antes ou igual à data da transação
    dados_finais = serie_hist.loc[serie_hist.index.to_period('M') <= data_transacao_period].iloc[-1]
    
    # Criar um dicionário com as variáveis calculadas
    variaveis_macroeconomicas = {
        'ipca_soma_12m': dados_finais['ipca_soma_12m'],
        'cambio_media_6m': dados_finais['cambio_media_6m'],
        'indicefipezap': dados_finais['indicefipezap'],
        'selic_media_6m': dados_finais['selic_media_6m'],
        'igpm_soma_12m': dados_finais['igpm_soma_12m'],
        'inpc_soma_12m': dados_finais['inpc_soma_12m']
    }

    return variaveis_macroeconomicas

# Função para incorporar variáveis macroeconômicas no DataFrame `dados_calculados`
def adicionar_variaveis_macroeconomicas(data_transacao, dados_calculados):
    # Calcular as variáveis macroeconômicas
    variaveis_macroeconomicas = calcular_variaveis_macroeconomicas(data_transacao)
    
    # Adicionar as variáveis calculadas no DataFrame `dados_calculados`
    for chave, valor in variaveis_macroeconomicas.items():
        dados_calculados[chave] = valor
    
    return dados_calculados


def  main_prep_variables(df_imoveis,
                        col_lat,
                        col_lon,
                        col_ano_construcao,
                        col_fracao_ideal,
                        col_area_construida,
                        col_area_terreno,
                        data_transacao):
    # Carregar os GeoDataFrames de referência
    gdf_parque, gdf_escolas, gdf_metro, gdf_hospital, gdf_shopping = carregar_gdfs_referencia()
    
    # Carregar o GeoDataFrame de treinamento (assumindo que ele foi salvo anteriormente)
    df_dados_treino = pd.read_pickle(path_dados_dataprep)
    
    # Transformar o DataFrame de treinamento em um GeoDataFrame
    df_dados_treino['geometry'] = df_dados_treino.apply(lambda row: Point(row['lon_goog'], row['lat_goog']), axis=1)
    gdf_dados_treino = gpd.GeoDataFrame(df_dados_treino, geometry='geometry', crs="EPSG:4326")
    
    # Converter o sistema de coordenadas para EPSG:32723
    gdf_dados_treino = gdf_dados_treino.to_crs(epsg=32723)

    
    
    df_imoveis2 = adicionar_variaveis_macroeconomicas(data_transacao, df_imoveis)
    # df_imoveis2 = df_imoveis.copy(deep=True)
    
    
    # Supondo que seu DataFrame inicial seja df_imoveis e contém as colunas necessárias como 'lat', 'lon', etc.
    def calcular_variaveis_para_dataframe(df,
                                          col_lat,
                                          col_lon,
                                          col_ano_construcao,
                                          col_fracao_ideal,
                                          col_area_construida,
                                          col_area_terreno,
                                          data_transacao,
                                          gdf_parque, gdf_metro, gdf_escolas, gdf_hospital, gdf_shopping, gdf_dados_treino):
        # DataFrame para armazenar os resultados
        lista_variaveis = []
        
        # Itera sobre cada linha do DataFrame original
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculando variáveis"):
            variaveis_calculadas = calcular_variaveis(
                lat=row[col_lat],
                lon=row[col_lon],
                ano_construcao = row[col_ano_construcao],
                area_terreno=row[col_area_terreno],
                fracao_ideal=row[col_fracao_ideal],
                area_construida=row[col_area_construida],
                gdf_parque=gdf_parque,
                gdf_metro=gdf_metro,
                gdf_escolas=gdf_escolas,
                gdf_hospital=gdf_hospital,
                gdf_shopping=gdf_shopping,
                gdf_dados_treino=gdf_dados_treino
            )
            lista_variaveis.append(variaveis_calculadas.iloc[0])  # Adiciona o resultado da linha atual
        
        # Concatena o DataFrame original com o novo DataFrame de variáveis calculadas
        df_resultado = pd.concat([df.reset_index(drop=True), pd.DataFrame(lista_variaveis).reset_index(drop=True)], axis=1)
        return df_resultado


    # Aplicação da função ao DataFrame
    df_imoveis_calculado = calcular_variaveis_para_dataframe(
        df_imoveis2,
        col_lat,
        col_lon,
        col_ano_construcao,
        col_fracao_ideal,
        col_area_construida,
        col_area_terreno,
        data_transacao,
        gdf_parque, 
        gdf_metro, 
        gdf_escolas, 
        gdf_hospital, 
        gdf_shopping, 
        gdf_dados_treino)


    # df_imoveis_calculado = df_resultado
    # df_imoveis_calculado['indicefipezap'] = 264.470658673279
    # df_imoveis_calculado['cambio_media_6m'] = 5.5749
    # df_imoveis_calculado['selic_media_6m'] = 0.1175
    # df_imoveis_calculado['ipca_soma_12m'] = 0.0434
    # df_imoveis_calculado['igpm_soma_12m'] = 0.0444
    # df_imoveis_calculado['inpc_soma_12m'] = 0.0402	


    # =============================================================================
    # Previsão modelo
    # =============================================================================
    
    # Função para carregar o modelo salvo e realizar previsões
    def carregar_modelo_e_prever(dados_imovel, modelo_path):
        # Carregar o modelo salvo com joblib
        modelo = joblib.load(modelo_path)
        feature_names = modelo.feature_names_in_
        dados_imovel = dados_imovel[feature_names]
        
        
        # # Realizar a previsão com os dados fornecidos
        previsao = modelo.predict(dados_imovel)
        
        
        # # Retornar o valor previsto
        return previsao
    
    # colunas = ['área do terreno (m2)', 'fração ideal', 'área construída (m2)', 'idade', 'cambio_media_6m', 'selic_media_6m', 'ipca_soma_12m', 'igpm_soma_12m', 'inpc_soma_12m', 'lat_goog', 'lon_goog', 'escolas_priv_1km', 'escolas_priv_2km', 'escolas_pub_1km', 'escolas_pub_2km', 'metro_1km', 'metro_2km', 'metro_3km', 'trem_1km', 'trem_2km', 'trem_3km', 'hospitais_1km', 'hospitais_2km', 'hospitais_3km', 'UBS_1km', 'UBS_2km', 'UBS_3km', 'distancia_shopping_mais_proximo', 'indicefipezap', 'media_preco_10_mais_proximos', 'mediana_preco_10_mais_proximos', 'media_preco_20_mais_proximos', 'mediana_preco_20_mais_proximos', 'media_preco_30_mais_proximos', 'mediana_preco_30_mais_proximos']
    
    # df_imoveis_calculado['lat_goog'] = pd.to_numeric(df_imoveis_calculado['lat_goog'], errors='coerce')
    # df_imoveis_calculado['lon_goog'] = pd.to_numeric(df_imoveis_calculado['lon_goog'], errors='coerce')
    
    
    df_imoveis_calculado['vlr_predito'] = carregar_modelo_e_prever(df_imoveis_calculado, modelo_path)
    
    return df_imoveis_calculado
# df_imoveis_calculado.to_excel('resultado_v3.xlsx')
