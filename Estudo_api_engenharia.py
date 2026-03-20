#%%
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# A MÁGICA PARA A NUVEM: O Python descobre sozinho a pasta onde ele está guardado!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, "modelo_concreto.pkl")

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

# Caminho absoluto para evitar problemas de localização do ficheiro
CAMINHO_MODELO = r"C:\Users\saulo\Downloads\estudo_exploratorio\modelo_concreto.pkl"

# 2. Carregamento do Modelo Treinado
try:
    modelo = joblib.load(CAMINHO_MODELO)
    print(f"✅ Modelo XGBoost carregado com sucesso a partir de: {CAMINHO_MODELO}")
except Exception as e:
    modelo = None
    print(f"❌ Erro: Modelo não encontrado em {CAMINHO_MODELO}. Detalhe: {e}")

# 3. Tabela de Custos (R$ por kg) - Pode ajustar conforme os preços do mercado local
CUSTOS_KG = {
    "cement": 0.50,
    "slag": 0.20,
    "fly_ash": 0.15,
    "water": 0.01,
    "superplasticizer": 3.00,
    "coarse_agg": 0.05,
    "fine_agg": 0.05
}

# ---------------------------------------------------------
# ESQUEMAS DE VALIDAÇÃO (Isto impede que o utilizador envie texto em vez de números)
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

# ---------------------------------------------------------
# ROTAS DA API
# ---------------------------------------------------------

@app.get("/", tags=["Health Check"])
def home():
    return {"status": "Online", "mensagem": "A API Concreto Inteligente está a funcionar na perfeição! 🚀"}

@app.post("/simular_traco", tags=["1. Validador de Traço (Engenharia)"])
def simular_traco(mix: MixConcreto):
    """
    **O que faz:** O utilizador insere uma receita de concreto e a IA prevê a Resistência e o Custo Total.
    """
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado no servidor.")

    dados = mix.model_dump()
    
    # Aplica a Engenharia de Features que a IA espera
    dados["water_cement_ratio"] = dados["water"] / dados["cement"]
    
    # Organiza na ordem exata em que o XGBoost foi treinado
    df_pred = pd.DataFrame([dados])
    colunas_modelo = ["cement", "slag", "fly_ash", "water", "superplasticizer", 
                      "coarse_agg", "fine_agg", "age_days", "water_cement_ratio"]
    df_pred = df_pred[colunas_modelo]
    
    # A Mágica: A IA faz a previsão
    resistencia_estimada = float(modelo.predict(df_pred)[0])
    
    # A Conta: Calcula o custo total
    custo_total = sum(dados[k] * CUSTOS_KG.get(k, 0) for k in ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"])
    
    return {
        "aviso": "Previsão baseada no modelo XGBoost otimizado",
        "resultados": {
            "resistencia_esperada_mpa": round(resistencia_estimada, 2),
            "custo_estimado_reais_por_m3": round(custo_total, 2)
        },
        "receita_enviada": dados
    }

@app.get("/otimizar_custo", tags=["2. Otimizador de Custo (Financeiro)"])
def otimizar_custo(resistencia_alvo_mpa: float = 30.0, num_opcoes: int = 3):
    """
    **O que faz:** O utilizador pede "Preciso de um concreto de 30 MPa" e a IA corre milhares de simulações 
    para encontrar as receitas MAIS BARATAS que garantem essa resistência.
    """
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado.")

    simulacoes = []
    
    # Simulação Monte Carlo rápida para a API responder num instante
    for _ in range(5000):
        mix = {
            "cement": float(np.random.uniform(100, 500)),
            "slag": float(np.random.uniform(0, 200)),
            "fly_ash": float(np.random.uniform(0, 200)),
            "water": float(np.random.uniform(120, 220)), # Restrição na água para evitar traços inviáveis
            "superplasticizer": float(np.random.uniform(0, 15)),
            "coarse_agg": float(np.random.uniform(850, 1150)),
            "fine_agg": float(np.random.uniform(650, 950)),
            "age_days": 28.0 # Fixo em 28 dias para padrão comercial
        }
        
        mix["water_cement_ratio"] = mix["water"] / mix["cement"]
        
        # O detalhe de engenharia: rejeitar misturas fisicamente ilógicas (ex: muito mais areia que brita)
        proporcao_agregados = mix["coarse_agg"] / mix["fine_agg"]
        if 1.0 <= proporcao_agregados <= 2.0: 
            custo = sum(mix[k] * CUSTOS_KG[k] for k in ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"])
            mix["custo_reais"] = custo
            simulacoes.append(mix)

    if not simulacoes:
        return {"erro": "A simulação não conseguiu gerar traços fisicamente viáveis. Tente novamente."}

    # Prever todas as milhares de opções de uma vez só (muito mais eficiente)
    df_sim = pd.DataFrame(simulacoes)
    colunas_modelo = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "water_cement_ratio"]
    df_sim["resistencia_estimada_mpa"] = modelo.predict(df_sim[colunas_modelo])
    
    # Filtra apenas as que atingem a meta pedida
    df_viavel = df_sim[df_sim["resistencia_estimada_mpa"] >= resistencia_alvo_mpa].copy()
    
    if df_viavel.empty:
        return {"mensagem": f"Não encontrámos traços viáveis para atingir {resistencia_alvo_mpa} MPa com os limites atuais de segurança."}
    
    # Ordena pelo custo (do mais barato para o mais caro)
    df_viavel = df_viavel.sort_values(by="custo_reais", ascending=True)
    
    # Seleciona as melhores opções
    melhores = df_viavel.head(num_opcoes).to_dict(orient="records")
    
    # Arredonda tudo para ficar apresentável no ecrã
    for m in melhores:
        for k, v in m.items():
            m[k] = round(v, 2)
            
    return {
        "objetivo": f"Encontrar traços de, no mínimo, {resistencia_alvo_mpa} MPa (Cura de 28 dias)",
        "opcoes_mais_baratas": melhores
    }

# NOVA ROTA PARA FORNECER DADOS REAIS DO CSV PARA O DASHBOARD
@app.get("/dados_graficos", tags=["3. Dados para Gráficos (Dashboard)"])
def dados_graficos():
    """
    Retorna os dados estatísticos reais do CSV para o HTML desenhar os gráficos interativos.
    """
    try:
        # Estratégia "Caçador": Vai testar todos os caminhos possíveis até achar o seu CSV
        caminhos_possiveis = [
            r"C:\Users\saulo\Downloads\estudo_exploratorio\dados\Estudo_exploratorio.csv",
            r"C:\Users\saulo\Downloads\estudo_exploratorio\Estudo_exploratorio.csv",
            r"C:\Users\saulo\Downloads\Estudo_exploratorio.csv",
            "dados/Estudo_exploratorio.csv",
            "Estudo_exploratorio.csv"
        ]
        
        df = None
        for caminho in caminhos_possiveis:
            if os.path.exists(caminho):
                df = pd.read_csv(caminho)
                print(f"✅ CSV encontrado com sucesso em: {caminho}")
                break
                
        if df is None:
            return {"erro": "O arquivo Estudo_exploratorio.csv não foi encontrado em nenhuma pasta."}
        
        # 1. Dados para o Histograma (Distribuição)
        distribuicao = df["compressive_strength_mpa"].tolist()
        
        # 2. Dados para a Matriz de Correlação
        cols_corr = ["cement", "water", "coarse_agg", "fine_agg", "age_days", "compressive_strength_mpa"]
        nomes_bonitos = ["Cimento", "Água", "Brita", "Areia", "Idade", "Resistência"]
        df_corr = df[cols_corr]
        corr_matrix = df_corr.corr().round(2).values.tolist()
        
        # 3. Importância das Variáveis (O que o XGBoost aprendeu)
        if modelo is not None:
            importancias = [float(i) for i in modelo.feature_importances_]
            nomes_features = ["Cimento", "Escória", "Cinza", "Água", "Aditivo", "Brita", "Areia", "Idade", "Fator A/C"]
        else:
            importancias = []
            nomes_features = []

        return {
            "distribuicao": distribuicao,
            "correlacao": {
                "valores": corr_matrix,
                "nomes": nomes_bonitos
            },
            "importancia": {
                "valores": importancias,
                "nomes": nomes_features
            }
        }
    except Exception as e:
        return {"erro": f"Erro interno ao ler os dados: {e}"}

@app.get("/amostra_dados", tags=["4. Dados Brutos"])
def amostra_dados():
    """Retorna as 10 primeiras linhas do dataset com o custo calculado."""
    try:
        caminho_csv = os.path.join(BASE_DIR, "Estudo_exploratorio.csv")
        if os.path.exists(caminho_csv):
            df = pd.read_csv(caminho_csv)
            
            # Tabela de custos (a mesma do otimizador)
            CUSTOS_KG = {
                "cement": 0.50, "slag": 0.20, "fly_ash": 0.15, "water": 0.01,
                "superplasticizer": 3.00, "coarse_agg": 0.05, "fine_agg": 0.05
            }
            
            # Calcula o custo da receita na hora (Matemática de matrizes do Pandas)
            df['custo_reais'] = (
                df['cement'] * CUSTOS_KG['cement'] + df['slag'] * CUSTOS_KG['slag'] +
                df['fly_ash'] * CUSTOS_KG['fly_ash'] + df['water'] * CUSTOS_KG['water'] +
                df['superplasticizer'] * CUSTOS_KG['superplasticizer'] +
                df['coarse_agg'] * CUSTOS_KG['coarse_agg'] + df['fine_agg'] * CUSTOS_KG['fine_agg']
            )
            
            # Pega as 10 primeiras linhas e transforma em dicionário
            amostra = df.sample(10).round(2).to_dict(orient="records")
            return {"amostra": amostra}
        return {"erro": "Arquivo não encontrado na nuvem."}
    except Exception as e:
        return {"erro": str(e)}

@app.get("/download_csv", tags=["4. Dados Brutos"])
def download_csv():
    """Permite o download do arquivo CSV original."""
    caminho_csv = os.path.join(BASE_DIR, "Estudo_exploratorio.csv")
    if os.path.exists(caminho_csv):
        # Envia o arquivo forçando o download no navegador
        return FileResponse(path=caminho_csv, filename="Super_Dataset_Concreto.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Arquivo CSV não encontrado.")
#%%