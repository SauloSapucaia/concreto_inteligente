#%%
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

#%%
# A MÁGICA DA NUVEM: O Python acha a própria pasta sozinho
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, "modelo_concreto.pkl")
CAMINHO_CSV = os.path.join(BASE_DIR, "Estudo_exploratorio.csv")

app = FastAPI(
    title="API Concreto Inteligente 🚀",
    description="Simulador e Otimizador de Traços",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#%%
# Carrega o Modelo
try:
    modelo = joblib.load(CAMINHO_MODELO)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    modelo = None
    print(f"❌ Erro ao carregar modelo: {e}")
#%%
CUSTOS_KG = {
    "cement": 0.50, "slag": 0.20, "fly_ash": 0.15, "water": 0.01,
    "superplasticizer": 3.00, "coarse_agg": 0.05, "fine_agg": 0.05
}
#%%
class MixConcreto(BaseModel):
    cement: float = Field(default=350.0)
    slag: float = Field(default=0.0)
    fly_ash: float = Field(default=0.0)
    water: float = Field(default=180.0)
    superplasticizer: float = Field(default=0.0)
    coarse_agg: float = Field(default=1000.0)
    fine_agg: float = Field(default=800.0)
    age_days: float = Field(default=28.0)

#%%
@app.get("/", tags=["Health Check"])
def home():
    return {"status": "Online", "mensagem": "API a funcionar! 🚀"}

#%%
@app.post("/simular_traco", tags=["1. Validador"])
def simular_traco(mix: MixConcreto):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado.")

    dados = mix.model_dump()
    dados["water_cement_ratio"] = dados["water"] / dados["cement"]
    
    df_pred = pd.DataFrame([dados])
    colunas_modelo = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "water_cement_ratio"]
    df_pred = df_pred[colunas_modelo]
    
    resistencia_estimada = float(modelo.predict(df_pred)[0])
    custo_total = sum(dados[k] * CUSTOS_KG.get(k, 0) for k in ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"])
    
    return {
        "resultados": {
            "resistencia_esperada_mpa": round(resistencia_estimada, 2),
            "custo_estimado_reais_por_m3": round(custo_total, 2)
        }
    }

#%%
@app.get("/otimizar_custo", tags=["2. Otimizador"])
def otimizar_custo(resistencia_alvo_mpa: float = 30.0, num_opcoes: int = 1):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado.")

    simulacoes = []
    for _ in range(5000):
        mix = {
            "cement": float(np.random.uniform(100, 500)),
            "slag": float(np.random.uniform(0, 200)),
            "fly_ash": float(np.random.uniform(0, 200)),
            "water": float(np.random.uniform(120, 220)),
            "superplasticizer": float(np.random.uniform(0, 15)),
            "coarse_agg": float(np.random.uniform(850, 1150)),
            "fine_agg": float(np.random.uniform(650, 950)),
            "age_days": 28.0
        }
        mix["water_cement_ratio"] = mix["water"] / mix["cement"]
        
        proporcao_agregados = mix["coarse_agg"] / mix["fine_agg"]
        if 1.0 <= proporcao_agregados <= 2.0: 
            custo = sum(mix[k] * CUSTOS_KG[k] for k in ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"])
            mix["custo_reais"] = custo
            simulacoes.append(mix)

    if not simulacoes:
        return {"erro": "Falha na simulação."}

    df_sim = pd.DataFrame(simulacoes)
    colunas_modelo = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "water_cement_ratio"]
    df_sim["resistencia_estimada_mpa"] = modelo.predict(df_sim[colunas_modelo])
    
    df_viavel = df_sim[df_sim["resistencia_estimada_mpa"] >= resistencia_alvo_mpa].copy()
    
    if df_viavel.empty:
        return {"erro": "Não encontramos traços."}
    
    df_viavel = df_viavel.sort_values(by="custo_reais", ascending=True)
    melhores = df_viavel.head(num_opcoes).to_dict(orient="records")
    
    for m in melhores:
        for k, v in m.items():
            m[k] = round(v, 2)
            
    return {"opcoes_mais_baratas": melhores}

#%%
@app.get("/dados_graficos", tags=["3. Dashboard"])
def dados_graficos():
    try:
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
            nomes_features = ["Cimento", "Escória", "Cinza", "Água", "Aditivo", "Brita", "Areia", "Idade", "Fator A/C"]
        else:
            importancias = []
            nomes_features = []

        return {
            "distribuicao": distribuicao,
            "correlacao": {"valores": corr_matrix, "nomes": nomes_bonitos},
            "importancia": {"valores": importancias, "nomes": nomes_features}
        }
    except Exception as e:
        return {"erro": str(e)}

#%%
@app.get("/amostra_dados", tags=["4. Tabela"])
def amostra_dados():
    try:
        if os.path.exists(CAMINHO_CSV):
            df = pd.read_csv(CAMINHO_CSV)
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
#%%
@app.get("/download_csv")
def download_csv():
    if os.path.exists(CAMINHO_CSV):
        return FileResponse(path=CAMINHO_CSV, filename="Super_Dataset_Concreto.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Arquivo não encontrado.")

#%%
