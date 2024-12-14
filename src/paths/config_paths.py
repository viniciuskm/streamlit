from pathlib import Path


# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# Data Dirs
# =============================================================================
# Definição dos diretórios principais
DATA_DIR = BASE_DIR / 'data'
p_data_links = BASE_DIR / 'data' / 'links'
# LOGS_DIR = DATA_DIR / 'logs' / 'leiloeiros'
p_data_processed = DATA_DIR / 'processed'
p_data_raw = DATA_DIR / 'raw'
p_data_consolidated_prepgpt = DATA_DIR / 'consolidated' / 'prep_gpt'
p_data_consolidated_output_gpt = DATA_DIR / 'consolidated' / 'output_gpt'
p_data_consolidated_prep_model = DATA_DIR / 'consolidated' / 'prep_model'

p_data_consolidated_predicoes = DATA_DIR / 'consolidated' / 'predicted'
p_data_consolidated_faltou_info = DATA_DIR / 'consolidated' / 'predicted' / 'info_incompleta'


p_dataprep_model_full = DATA_DIR.parent.parent / 'previsao_precos_imoveis_model' / 'data' / 'processed' / 'prep_model' / 'dataprep_full_v2.pkl'




# =============================================================================
# Model Dirs
# =============================================================================
MODEL_DIR = BASE_DIR / 'models'

p_model_xgb_fullvariables = MODEL_DIR / '01_arquivofunctionprep' / 'best_xgb_model.pkl'
p_model_dataprep = MODEL_DIR / '01_arquivofunctionprep' / 'data' / 'dataprep_model_v2.pkl'
p_model_equipamentos_saopaulo = MODEL_DIR / '01_arquivofunctionprep' / 'data' / '01_equipamentos_saopaulo'
p_model_equipamentos_fipezap = MODEL_DIR / '01_arquivofunctionprep' / 'data' / 'fipezap' / 'fipezap-serieshistoricas.xlsx'


# function_dataprep = BASE_DIR.parent / 'previsao_precos_imoveis_model' / 'src'/ 'data'



# RAW_DATA_DIR = DATA_DIR / 'raw' / 'leiloeiros'
# SHAPE_DIR = DATA_DIR / 'raw' / 'shape_saopaulo'
# CONSOLIDATED_DATA_DIR = DATA_DIR / 'consolidated'

# Exemplo de caminho para um arquivo específico
# EXAMPLE_LOG_FILE = LOGS_DIR / 'example_log.txt'

# Criação das pastas, se não existirem
# LOGS_DIR.mkdir(parents=True, exist_ok=True)
# PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
# RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# SHAPE_DIR.mkdir(parents=True, exist_ok=True)
# CONSOLIDATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
