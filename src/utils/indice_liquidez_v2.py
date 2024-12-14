import pandas as pd
from shapely.geometry import Point
from geopandas import gpd
from scipy.spatial import KDTree
import numpy as np
from tqdm import tqdm

# df = pd.read_excel('../data/leiloes_teste/resultado_v5.xlsx')

# df = pd.read_excel('../data/leiloes_teste/leiloes_predicted_2024-11-09_to_2024-11-10 (1).xlsx')
# df = pd.read_excel('../data/leiloes_teste/novos_leiloes_chris.xlsx')

# df = pd.read_excel(r'C:\Users\Vinicius_Maia\OneDrive - Insper - Institudo de Ensino e Pesquisa\Documents\Vinicius\Leiloes_v2\leilao_scraper_project\teste.xlsx')


# Definindo os valores de latitude e longitude
# lat_lon = {'latitude': [-23.61265489267088], 'longitude': [-46.62554273225787]}

# Criando o DataFrame
# df = pd.DataFrame(lat_lon)



def classificar_liquidez_clusters(gdf, fronteiras):
    """Classifica o índice de liquidez em clusters com base nas fronteiras fornecidas."""
    
    def atribuir_cluster(valor):
        for cluster, (limite_inferior, limite_superior) in fronteiras.items():
            if limite_inferior <= valor < limite_superior:
                return cluster
        return None  # Caso o valor não se enquadre em nenhuma fronteira

    # Aplica a classificação por cluster usando a função `atribuir_cluster`
    gdf['liquidez_cluster'] = gdf['indice_liquidez'].apply(atribuir_cluster)





def calcular_indice_liquidez_otimizado(
    df, 
    raio=200, 
    dist_empreendimento=10, 
    pesos_anos={2024: 2.0, 2023: 1.5, 2022: 1.2}, 
    peso_extra_empreendimento=1.2
):
    """Calcula o índice de liquidez para cada imóvel em gdf_test com base nos dados de gdf."""
    
    # Convertendo para GeoDataFrame com pontos geométricos
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf_test = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Convertendo o sistema de coordenadas para métrico (UTM)
    gdf_test = gdf_test.to_crs(epsg=32723)

    # Definição das fronteiras dos clusters para classificação
    fronteiras_clusters = {
        1: (0.00, 156),
        2: (156, 272),
        3: (272, 402),
        4: (402, 585),
        5: (585, float('inf'))
    }

    # Carregando o GeoDataFrame de referência
    gdf = pd.read_pickle('tests/01_indice_liquidez/outputs/dataprep_full_indice_liquidez_20241207.pkl')

    # Extrai coordenadas e anos de venda para o gdf
    coords_gdf = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    anos_venda_gdf = gdf['ano'].values

    # Constrói o KDTree para o gdf (banco de dados)
    tree = KDTree(coords_gdf)
    
    # Inicializa lista para armazenar os índices de liquidez para gdf_test
    liquidez_resultados = []

    # Itera sobre cada imóvel em gdf_test
    coords_test = np.array(list(zip(gdf_test.geometry.x, gdf_test.geometry.y)))
    for idx, (x, y) in tqdm(enumerate(coords_test), total=len(gdf_test), desc="Calculando índice de liquidez para gdf_test"):
        try:
            liquidez = 0

            # Busca todos os imóveis em gdf dentro do raio especificado
            indices_proximos = tree.query_ball_point((x, y), raio, p=2)
            
            # Itera sobre os índices dos imóveis próximos em gdf
            for i in indices_proximos:
                # Verifica o ano da venda e aplica o peso temporal
                ano_venda = anos_venda_gdf[i]
                peso_tempo = pesos_anos.get(ano_venda, 1.0)  # Peso padrão de 1.0 se o ano não estiver em pesos_anos
                
                # Calcula a distância e aplica o peso adicional se for o mesmo empreendimento
                distancia = np.linalg.norm([x - coords_gdf[i][0], y - coords_gdf[i][1]])
                peso_espacial = peso_extra_empreendimento if distancia <= dist_empreendimento else 1.0
                
                # Contribuição de liquidez para o imóvel vizinho com pesos
                liquidez += peso_tempo * peso_espacial
            
            # Armazena o índice de liquidez calculado para o imóvel atual em gdf_test
            liquidez_resultados.append(liquidez)
        except Exception as e:
            print(f"Erro no ponto índice {idx} - Coordenadas: ({x}, {y}) - Erro: {e}")
            liquidez_resultados.append(np.nan)  # Marca como NaN em caso de erro
    
    # Adiciona a coluna de liquidez ao gdf_test
    gdf_test['indice_liquidez'] = liquidez_resultados
    
    # Classifica o índice de liquidez em clusters usando as fronteiras
    classificar_liquidez_clusters(gdf_test, fronteiras_clusters)
    
    return gdf_test[['indice_liquidez', 'liquidez_cluster']]


def calcular_indice_liquidez_otimizado_v2(
    gdf_test,
    gdf_prep_full,
    raio=200, 
    dist_empreendimento=10, 
    pesos_anos={2024: 2.0, 2023: 1.5, 2022: 1.2}, 
    peso_extra_empreendimento=1.2,
    # crs_certo=1
):
    """Calcula o índice de liquidez para cada imóvel em gdf_test com base nos dados de gdf."""
    
    # if crs_certo!=1:
    # Convertendo para GeoDataFrame com pontos geométricos
        # geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        # gdf_test = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        # Convertendo o sistema de coordenadas para métrico (UTM)
        # gdf_test = gdf_test.to_crs(epsg=32723)
    
    # Definição das fronteiras dos clusters para classificação
    fronteiras_clusters = {
        1: (0.00, 156),
        2: (156, 272),
        3: (272, 402),
        4: (402, 585),
        5: (585, float('inf'))
    }

    # Carregando o GeoDataFrame de referência
    # gdf = pd.read_pickle('tests/01_indice_liquidez/outputs/dataprep_full_indice_liquidez_20241207.pkl')

    # Extrai coordenadas e anos de venda para o gdf
    coords_gdf = np.array(list(zip(gdf_prep_full.geometry.x, gdf_prep_full.geometry.y)))
    anos_venda_gdf = gdf_prep_full['data de transação'].dt.year.values
    # anos_venda_gdf = gdf_prep_full['ano'].values

    # Constrói o KDTree para o gdf (banco de dados)
    tree = KDTree(coords_gdf)
    
    # Inicializa lista para armazenar os índices de liquidez para gdf_test
    liquidez_resultados = []

    # Itera sobre cada imóvel em gdf_test
    coords_test = np.array(list(zip(gdf_test.geometry.x, gdf_test.geometry.y)))
    for idx, (x, y) in tqdm(enumerate(coords_test), total=len(gdf_test), desc="Calculando índice de liquidez para gdf_test"):
        try:
            liquidez = 0

            # Busca todos os imóveis em gdf dentro do raio especificado
            indices_proximos = tree.query_ball_point((x, y), raio, p=2)
            
            # Itera sobre os índices dos imóveis próximos em gdf
            for i in indices_proximos:
                # Verifica o ano da venda e aplica o peso temporal
                ano_venda = anos_venda_gdf[i]
                peso_tempo = pesos_anos.get(ano_venda, 1.0)  # Peso padrão de 1.0 se o ano não estiver em pesos_anos
                
                # Calcula a distância e aplica o peso adicional se for o mesmo empreendimento
                distancia = np.linalg.norm([x - coords_gdf[i][0], y - coords_gdf[i][1]])
                peso_espacial = peso_extra_empreendimento if distancia <= dist_empreendimento else 1.0
                
                # Contribuição de liquidez para o imóvel vizinho com pesos
                liquidez += peso_tempo * peso_espacial
            
            # Armazena o índice de liquidez calculado para o imóvel atual em gdf_test
            liquidez_resultados.append(liquidez)
        except Exception as e:
            print(f"Erro no ponto índice {idx} - Coordenadas: ({x}, {y}) - Erro: {e}")
            liquidez_resultados.append(np.nan)  # Marca como NaN em caso de erro
    
    # Adiciona a coluna de liquidez ao gdf_test
    gdf_test['indice_liquidez'] = liquidez_resultados
    
    # Classifica o índice de liquidez em clusters usando as fronteiras
    classificar_liquidez_clusters(gdf_test, fronteiras_clusters)
    
    return gdf_test[['indice_liquidez', 'liquidez_cluster']]




# gdf[gdf.indice_liquidez<=100].shape


# gdf_test.head()
# resultado = calcular_indice_liquidez_otimizado(df)
# print(resultado)
# gdf_test.to_excel('resultado_v6.xlsx')


# print('\n',gdf_test[['indice_liquidez', 'liquidez_cluster']])
