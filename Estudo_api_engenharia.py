#%%
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from scipy.stats import norm

# A MÁGICA PARA A NUVEM: O Python descobre sozinho a pasta onde ele está guardado!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, "modelo_concreto.pkl")
CAMINHO_CSV = os.path.join(BASE_DIR, "Estudo_exploratorio.csv")

# 1. Inicialização da API
app = FastAPI(
    title="API Concreto Inteligente 🚀",
    description="Simulador e Otimizador de Traços de Betão/Concreto baseados em Machine Learning (XGBoost).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Carregamento do Modelo Treinado
try:
    modelo = joblib.load(CAMINHO_MODELO)
    print(f"✅ Modelo XGBoost carregado com sucesso!")
except Exception as e:
    modelo = None
    print(f"❌ Erro ao carregar modelo: {e}")

# ---------------------------------------------------------
# 3. Tabela de Custos REAIS (R$ por kg) - Mercado Brasileiro
# ---------------------------------------------------------
CUSTOS_KG = {
    "cement": 0.85,             # ~ R$ 42,50 o saco de 50kg
    "slag": 0.25,               # Escória
    "fly_ash": 0.20,            # Cinza volante
    "water": 0.02,              # Custo de água tratada na usina
    "superplasticizer": 12.00,  # Aditivos são caros por kg
    "coarse_agg": 0.08,         # Brita (~ R$ 80 a tonelada)
    "fine_agg": 0.07            # Areia (~ R$ 70 a tonelada)
}

# ---------------------------------------------------------
# ESQUEMAS DE VALIDAÇÃO
# ---------------------------------------------------------
class MixConcreto(BaseModel):
    cement: float = Field(default=350.0, ge=100, le=500, description="Cimento (kg/m³)")
    slag: float = Field(default=0.0, ge=0, le=200, description="Escória (kg/m³)")
    fly_ash: float = Field(default=0.0, ge=0, le=200, description="Cinza Volante (kg/m³)")
    water: float = Field(default=180.0, ge=100, le=250, description="Água (kg/m³)")
    superplasticizer: float = Field(default=0.0, ge=0, le=30, description="Superplastificante (kg/m³)")
    coarse_agg: float = Field(default=1000.0, ge=800, le=1200, description="Agregado Graúdo/Brita (kg/m³)")
    fine_agg: float = Field(default=800.0, ge=600, le=1000, description="Agregado Miúdo/Areia (kg/m³)")
    age_days: float = Field(default=28.0, ge=1, le=365, description="Idade de Cura (Dias)")
    resistencia_alvo_mpa: float = Field(default=30.0, ge=1, le=100, description="Resistência Exigida (Alvo em MPa)")

# ---------------------------------------------------------
# ROTAS DA API
# ---------------------------------------------------------

@app.get("/", tags=["Health Check"])
def home():
    return {"status": "Online", "mensagem": "A API Concreto Inteligente está a funcionar na perfeição! 🚀"}

@app.post("/simular_traco", tags=["1. Validador de Traço (Engenharia)"])
def simular_traco(mix: MixConcreto):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado no servidor.")

    dados = mix.dict() if hasattr(mix, 'dict') else mix.model_dump()
    
    if dados["cement"] <= 0:
        raise HTTPException(status_code=400, detail="A quantidade de cimento deve ser maior que zero.")
        
    dados["water_cement_ratio"] = dados["water"] / dados["cement"]
    
    df_pred = pd.DataFrame([dados])
    
    colunas_modelo = [
        "cement", "slag", "fly_ash", "water_cement_ratio", "superplasticizer", "age_days"
    ]
    df_pred = df_pred[colunas_modelo]
    
    resistencia_estimada = float(modelo.predict(df_pred)[0])
    custo_total = sum(dados[k] * CUSTOS_KG.get(k, 0) for k in ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"])

    RMSE_MODELO = 4.80 

    # A função norm.cdf calcula a integral de -infinito até o Alvo
    probabilidade_falha = norm.cdf(dados["resistencia_alvo_mpa"], loc=resistencia_estimada, scale=RMSE_MODELO)
    risco_percentual = probabilidade_falha * 100

    return {
        "resultados": {
            "resistencia_esperada_mpa": round(resistencia_estimada, 2),
            "custo_estimado_reais_por_m3": round(custo_total, 2)
        }
    }

@app.get("/otimizar_custo", tags=["2. Otimizador de Custo (Financeiro)"])
def otimizar_custo(resistencia_alvo_mpa: float = 30.0, num_opcoes: int = 1):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado.")

    simulacoes = []
    
    # Simulação Monte Carlo Aumentada e Blindada
    for _ in range(15000):
        mix = {
            "cement": float(np.random.uniform(280, 500)), # NUNCA menos de 280kg para estrutura!
            "slag": float(np.random.uniform(0, 150)),
            "fly_ash": float(np.random.uniform(0, 150)),
            "water": float(np.random.uniform(140, 220)), 
            "superplasticizer": float(np.random.uniform(0, 10)),
            "coarse_agg": float(np.random.uniform(900, 1150)),
            "fine_agg": float(np.random.uniform(700, 950)),
            "age_days": 28.0 
        }
        
        mix["water_cement_ratio"] = mix["water"] / mix["cement"]
        proporcao_agregados = mix["coarse_agg"] / mix["fine_agg"]
        
        # LEI DE ABRAMS: Fator A/C tem de estar entre 0.35 e 0.65
        if 1.0 <= proporcao_agregados <= 2.0 and 0.35 <= mix["water_cement_ratio"] <= 0.65: 
            custo = sum(mix[k] * CUSTOS_KG[k] for k in ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"])
            mix["custo_reais"] = custo
            simulacoes.append(mix)

    if not simulacoes:
        return {"erro": "A simulação não conseguiu gerar traços fisicamente viáveis. Tente novamente."}

    df_sim = pd.DataFrame(simulacoes)
    
    colunas_modelo = [ "cement", "slag", "fly_ash", "water_cement_ratio", "superplasticizer", "age_days" ]
    df_sim["resistencia_estimada_mpa"] = modelo.predict(df_sim[colunas_modelo])
    
    df_viavel = df_sim[df_sim["resistencia_estimada_mpa"] >= resistencia_alvo_mpa].copy()
    
    if df_viavel.empty:
        return {"erro": f"Não encontrámos traços viáveis para atingir {resistencia_alvo_mpa} MPa com os limites atuais de segurança."}
    
    df_viavel = df_viavel.sort_values(by="custo_reais", ascending=True)
    melhores = df_viavel.head(num_opcoes).to_dict(orient="records")
    
    for m in melhores:
        for k, v in m.items():
            m[k] = round(v, 2)
            
    return {"opcoes_mais_baratas": melhores}

@app.get("/dados_graficos", tags=["3. Dados para Gráficos (Dashboard)"])
def dados_graficos():
    try:
        # Removida a estratégia "Caçador" (caminhos do Windows) que causava erros
        if not os.path.exists(CAMINHO_CSV):
            return {"erro": "Arquivo CSV não encontrado na nuvem."}
            
        df = pd.read_csv(CAMINHO_CSV)
        distribuicao = df["compressive_strength_mpa"].tolist()
        
        cols_corr = ["cement", "water", "coarse_agg", "fine_agg", "age_days", "compressive_strength_mpa"]
        nomes_bonitos = ["Cimento", "Água", "Brita", "Areia", "Idade", "Resistência"]
        df_corr = df[cols_corr]
        corr_matrix = df_corr.corr().round(2).values.tolist()
        
        if modelo is not None:
            importancias = [float(i) for i in modelo.feature_importances_]
            nomes_features = ["Cimento", "Escória", "Cinza Volante", "Fator A/C", "Aditivo", "Idade"]
        else:
            importancias = []
            nomes_features = []

        return {
            "distribuicao": distribuicao,
            "correlacao": {"valores": corr_matrix, "nomes": nomes_bonitos},
            "importancia": {"valores": importancias, "nomes": nomes_features}
        }
    except Exception as e:
        return {"erro": f"Erro interno ao ler os dados: {e}"}

@app.get("/amostra_dados", tags=["4. Dados Brutos"])
def amostra_dados():
    try:
        if os.path.exists(CAMINHO_CSV):
            df = pd.read_csv(CAMINHO_CSV)
            
            # Tabela de custos atualizada para a tabela também bater com os preços reais
            df['custo_reais'] = (
                df['cement'] * CUSTOS_KG['cement'] + df['slag'] * CUSTOS_KG['slag'] +
                df['fly_ash'] * CUSTOS_KG['fly_ash'] + df['water'] * CUSTOS_KG['water'] +
                df['superplasticizer'] * CUSTOS_KG['superplasticizer'] +
                df['coarse_agg'] * CUSTOS_KG['coarse_agg'] + df['fine_agg'] * CUSTOS_KG['fine_agg']
            )
            
            amostra = df.sample(10).round(2).to_dict(orient="records")
            return {"amostra": amostra}
        return {"erro": "Arquivo não encontrado na nuvem."}
    except Exception as e:
        return {"erro": str(e)}

@app.get("/download_csv", tags=["4. Dados Brutos"])
def download_csv():
    if os.path.exists(CAMINHO_CSV):
        return FileResponse(path=CAMINHO_CSV, filename="Super_Dataset_Concreto.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Arquivo CSV não encontrado.")
    
#%%