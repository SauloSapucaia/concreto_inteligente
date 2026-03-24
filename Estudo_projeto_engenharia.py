# %%
import os
import requests
import joblib
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("default")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from deap import base, creator, tools, algorithms
import shap

# %%
# 1. DOWNLOAD E PREPARAÇÃO DOS DADOS (SUPER DATASET)
# ==========================================
def preparar_dados():
    print("[1/5] Iniciando o carregamento dos dados...")
    os.makedirs("dados", exist_ok=True)
    
    # Atualizado para o caminho exato do seu computador
    pasta_kaggle = r"C:\Users\saulo\Downloads\estudo_exploratorio"
    
    caminho_arquivo = "dados/Concrete_Data.xls"
    url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

    if not os.path.exists(caminho_arquivo):
        print("      -> Baixando dataset UCI original...")
        r = requests.get(url_uci, timeout=30)
        r.raise_for_status()
        with open(caminho_arquivo, "wb") as f:
            f.write(r.content)
    
    # 1. Carrega a base padrão (UCI)
    df_uci = pd.read_excel(caminho_arquivo)
    df_uci.columns = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "compressive_strength_mpa"]
    
    frames = [df_uci]
    
    # 2. Varre a sua pasta de Downloads/estudo_exploratorio atrás dos novos arquivos
    if os.path.exists(pasta_kaggle):
        arquivos_extras = [f for f in os.listdir(pasta_kaggle) if f.endswith(('.csv', '.xls', '.xlsx'))]
        
        if arquivos_extras:
            print(f"      -> Encontrados {len(arquivos_extras)} arquivos extras na pasta. Iniciando padronização...")
            for arquivo in arquivos_extras:
                caminho = os.path.join(pasta_kaggle, arquivo)
                try:
                    if arquivo.endswith('.csv'):
                        try:
                            df_extra = pd.read_csv(caminho, sep=',')
                            if len(df_extra.columns) < 5: df_extra = pd.read_csv(caminho, sep=';')
                        except:
                            df_extra = pd.read_csv(caminho, sep=';', encoding='latin1')
                    else:
                        df_extra = pd.read_excel(caminho)
                    
                    colmap = {}
                    for c in df_extra.columns:
                        cs = str(c).lower()
                        if "cement" in cs or "cimento" in cs: colmap[c] = "cement"
                        elif "slag" in cs or "escoria" in cs: colmap[c] = "slag"
                        elif "fly" in cs or "cinza" in cs: colmap[c] = "fly_ash"
                        elif "water" in cs or "agua" in cs: colmap[c] = "water"
                        elif "super" in cs or "plast" in cs or "sp" == cs: colmap[c] = "superplasticizer"
                        elif "coarse" in cs or "graud" in cs or "brita" in cs: colmap[c] = "coarse_agg"
                        elif "fine" in cs or "miudo" in cs or "areia" in cs: colmap[c] = "fine_agg"
                        elif "age" in cs or "dias" in cs or "idade" in cs: colmap[c] = "age_days"
                        elif "strength" in cs or "compress" in cs or "resist" in cs or "mpa" in cs or "28_day_str" in cs: 
                            colmap[c] = "compressive_strength_mpa"
                    
                    df_extra = df_extra.rename(columns=colmap)
                    
                    if "age_days" not in df_extra.columns:
                        df_extra["age_days"] = 28.0 
                        
                    for aditivo in ["slag", "fly_ash", "superplasticizer"]:
                        if aditivo not in df_extra.columns:
                            df_extra[aditivo] = 0.0 
                    
                    colunas_necessarias = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age_days", "compressive_strength_mpa"]
                    
                    if all(col in df_extra.columns for col in colunas_necessarias):
                        for col in colunas_necessarias:
                            df_extra[col] = pd.to_numeric(df_extra[col], errors='coerce')
                        df_extra = df_extra[colunas_necessarias].dropna()
                        frames.append(df_extra)
                        print(f"         + Ingestão de '{arquivo}': {len(df_extra)} linhas extraídas com sucesso!")
                    else:
                        faltantes = [c for c in colunas_necessarias if c not in df_extra.columns]
                        print(f"         - Arquivo '{arquivo}' ignorado. Faltam as colunas principais: {faltantes}")
                        
                except Exception as e:
                    print(f"         x Erro ao ler '{arquivo}': {e}")
    else:
        print(f"      -> Aviso: A pasta {pasta_kaggle} não foi encontrada.")
                
    # 3. Junta tudo em um Super Dataset e remove duplicatas
    df_final = pd.concat(frames, ignore_index=True)
    linhas_antes = len(df_final)
    df_final = df_final.drop_duplicates()
    linhas_depois = len(df_final)
    
    if linhas_antes != linhas_depois:
        print(f"      -> Limpeza: {linhas_antes - linhas_depois} registros duplicados foram removidos.")
    
    print("      -> Criando nova feature: proporção água/cimento...")
    df_final["water_cement_ratio"] = df_final["water"] / df_final["cement"]
    
    # Agora com o caminho absoluto cravado!
    caminho_csv = r"C:\Users\saulo\Downloads\estudo_exploratorio\Estudo_exploratorio.csv"
    df_final.to_csv(caminho_csv, index=False)
    
    print(f"      -> SUCESSO! Super Dataset consolidado em: {caminho_csv}")
    print("-" * 60)
    
    return df_final

# %%
# 2. ANÁLISE EXPLORATÓRIA BÁSICA (EDA)
# ==========================================
def plotar_eda(df):
    print("\n[2/5] Gerando gráficos de Análise Exploratória (EDA)...")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df["compressive_strength_mpa"], kde=True, color='blue')
    plt.title("Distribuição da Resistência do Concreto (MPa)")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlação das Variáveis")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    colunas_insumos = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"]
    sns.boxplot(data=df[colunas_insumos], orient="h", palette="Set2")
    plt.title("Detecção de Outliers nas Quantidades de Insumos (kg/m³)")
    plt.xlabel("Quantidade (kg)")
    plt.show()
    
    print("      -> Gráficos gerados com sucesso.")
    print("-" * 60)

# %%
# 3. TREINAMENTO DOS MODELOS
# ==========================================
def treinar_modelos(df):
    print("\n[3/5] Iniciando o processamento, divisão e treinamento dos modelos...")
    
    if 'water_cement_ratio' not in df.columns:
        df['water_cement_ratio'] = df['water'] / df['cement']
    
    features_blindadas = [
        'cement', 
        'slag', 
        'fly_ash', 
        'water_cement_ratio', 
        'superplasticizer', 
        'age_days'
    ]
    
    X = df[features_blindadas]
    y = df["compressive_strength_mpa"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"      -> Dados divididos: {len(X_train)} linhas para treino, {len(X_test)} para teste.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("      -> Treinando Modelo de Baseline (Regressão Linear)...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    print("      -> Treinando Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train) 

    print("      -> Buscando os melhores parâmetros para o XGBoost (RandomizedSearch)...")
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb_base = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=param_grid, 
        n_iter=15, cv=5, scoring='neg_root_mean_squared_error', 
        verbose=0, random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    xgb_otimizado = random_search.best_estimator_
    print(f"      -> Melhor configuração encontrada: {random_search.best_params_}")

    print("      -> Calculando Validação Cruzada (K-Fold = 5 com embaralhamento) para o XGBoost...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_otimizado, X, y, cv=kf, scoring='r2')
    print(f"         Média R² no K-Fold: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\n      --- RESULTADOS DA AVALIAÇÃO (COMPARAÇÃO) ---")
    modelos_dict = {
        "Baseline (Regressão Linear)": (lr, X_test_scaled),
        "Random Forest": (rf, X_test),
        "XGBoost Otimizado": (xgb_otimizado, X_test)
    }

    for nome, (modelo, X_teste_usado) in modelos_dict.items():
        preds = modelo.predict(X_teste_usado)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds) 
        
        print(f"      [{nome}]")
        print(f"      R²   : {r2:.4f}")
        print(f"      RMSE : {rmse:.4f} MPa")
        print(f"      MAE  : {mae:.4f} MPa\n")

    print("      -> Gerando Inteligência Explicável (SHAP Values) para o Dashboard...")
    explainer = shap.TreeExplainer(xgb_otimizado)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.title("Impacto das Variáveis na Resistência do Concreto (Análise SHAP)", fontsize=14, pad=20)
    plt.tight_layout()
    nome_imagem = "shap_summary.png"
    plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
    plt.close()

    print("-" * 60)
    return xgb_otimizado, X.columns

# %%
# 4. OTIMIZAÇÃO MULTIOBJETIVO (NSGA-II)
# ==========================================
def checkBounds(min_val, max_val):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] < min_val[i]:
                        child[i] = min_val[i]
                    elif child[i] > max_val[i]:
                        child[i] = max_val[i]
            return offspring
        return wrapper
    return decorator

def otimizacao_genetica(modelo_xgb):
    print("\n[5/5] Iniciando otimização com Algoritmo Genético (DEAP)...")
    
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    custos = {
        "cement": 0.50, "slag": 0.20, "fly_ash": 0.15, "water": 0.01,
        "superplasticizer": 3.00, "coarse_agg": 0.05, "fine_agg": 0.05
    }

    def avaliar(individual):
        mix = {
            "cement": individual[0],
            "slag": individual[1],
            "fly_ash": individual[2],
            "water": individual[3],
            "superplasticizer": individual[4],
            "coarse_agg": individual[5],
            "fine_agg": individual[6],
            "age_days": 28 
        }
        mix["water_cement_ratio"] = mix["water"] / mix["cement"]
        df_temp = pd.DataFrame([mix])
        colunas_modelo = [ "cement", "slag", "fly_ash", "water_cement_ratio", "superplasticizer", "age_days" ]
        df_modelo = df_temp[colunas_modelo]
        resistencia = modelo_xgb.predict(df_modelo)[0]
        custo = sum(mix[k] * custos[k] for k in custos.keys())
        return custo, resistencia

    toolbox = base.Toolbox()
    
    MIN = [100, 0, 0, 100, 0, 800, 600, 7]
    MAX = [500, 200, 200, 250, 30, 1200, 1000, 56]
    
    toolbox.register("cement", random.uniform, MIN[0], MAX[0])
    toolbox.register("slag", random.uniform, MIN[1], MAX[1])
    toolbox.register("fly_ash", random.uniform, MIN[2], MAX[2])
    toolbox.register("water", random.uniform, MIN[3], MAX[3])
    toolbox.register("superplasticizer", random.uniform, MIN[4], MAX[4])
    toolbox.register("coarse_agg", random.uniform, MIN[5], MAX[5])
    toolbox.register("fine_agg", random.uniform, MIN[6], MAX[6])
    toolbox.register("age_days", random.uniform, MIN[7], MAX[7]) 

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.cement, toolbox.slag, toolbox.fly_ash, toolbox.water, 
                      toolbox.superplasticizer, toolbox.coarse_agg, toolbox.fine_agg, toolbox.age_days), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", avaliar)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    toolbox.decorate("mate", checkBounds(MIN, MAX))
    toolbox.decorate("mutate", checkBounds(MIN, MAX))

    pop = toolbox.population(n=100)
    
    print("      -> Evoluindo gerações (isso pode levar um tempo)...")
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=40, verbose=False
    )

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    custos_p = [ind.fitness.values[0] for ind in pareto]
    resistencias_p = [ind.fitness.values[1] for ind in pareto]

    print(f"      -> Otimização concluída! Foram encontradas {len(pareto)} soluções ótimas.")
    print(f"      -> Custo médio da fronteira: R$ {np.mean(custos_p):.2f} / m³")
    print(f"      -> Resistência média da fronteira: {np.mean(resistencias_p):.2f} MPa\n")

    print("      -> Gerando gráfico da Fronteira de Pareto...")
    plt.figure(figsize=(8, 5))
    plt.scatter(custos_p, resistencias_p, color='purple', alpha=0.7)
    plt.xlabel("Custo Estimado (R$ / m³)")
    plt.ylabel("Resistência Estimada (MPa)")
    plt.title("Fronteira de Pareto - Custo x Resistência (Traços Fisicamente Possíveis)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(left=max(0, min(custos_p) - 10)) 
    plt.show()

def explicar_modelo(modelo_xgb, X_train):
    print("\n[+] Gerando análise de explicabilidade SHAP...")
    
    # Explica as decisões da IA usando a Teoria dos Jogos
    explainer = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(X_train)
    
    # Gera o gráfico
    plt.figure(figsize=(10, 6))
    plt.title("O que mais impacta a Resistência do Concreto?")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig("explicabilidade_shap.png", bbox_inches='tight')
    plt.show()
    print("✅ Gráfico salvo como 'explicabilidade_shap.png'")

# %%
# 5. EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print(" INICIANDO PIPELINE DE CONCRETO INTELIGENTE ")
    print("==================================================\n")

    # 1. Preparar os dados
    df_concreto = preparar_dados()

    # 2. EDA (Descomente para ver os gráficos)
    plotar_eda(df_concreto)
    
    # 3. Treinar os modelos
    modelo_final, features = treinar_modelos(df_concreto)
    
    # 4. Salvar o modelo (AGORA COM CAMINHO FIXO)
    print("\n[4/5] Salvando o modelo treinado...")
    
    # Coloque o caminho exato de onde você quer que o arquivo apareça
    caminho_modelo = r"C:\Users\saulo\Downloads\estudo_exploratorio\modelo_concreto.pkl"
    joblib.dump(modelo_final, caminho_modelo)
    
    print(f"      -> Modelo XGBoost salvo com sucesso em: {caminho_modelo}")
    print("-" * 60)
    
    # 5. Otimização Genética
    otimizacao_genetica(modelo_final)

    print("\n==================================================")
    print(" PIPELINE FINALIZADO COM SUCESSO! ")
    print("==================================================")

#%%