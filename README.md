# 🏗️ Concreto Inteligente: Otimização de Traços com IA

> Aplicação de Machine Learning (XGBoost) e Algoritmos Genéticos (NSGA-II) para previsão de resistência e otimização de custos na Construção Civil. Projeto desenvolvido no ecossistema de Data Science da FIAP.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)

## 🌐 Acesso Online
O dashboard interativo e a API estão hospedados em nuvem e podem ser acessados diretamente pelo navegador:
👉 **[Clique aqui para aceder ao Simulador ao vivo](https://saulosapucaia.github.io/concreto_inteligente/)**

---

## 📌 Sobre o Projeto
O cálculo de traços de betão/concreto costuma depender de métodos empíricos e tabelas estáticas. Este projeto introduz uma abordagem orientada a dados, utilizando um modelo **Extreme Gradient Boosting (XGBoost)** treinado com mais de 1.000 amostras laboratoriais. 

O sistema conta com uma API robusta e um Dashboard interativo capaz de:
1. **Laboratório Virtual:** Prever a resistência (MPa) e o custo de uma mistura específica em milissegundos.
2. **Otimizador Financeiro:** Encontrar a combinação de materiais mais barata possível que atenda aos requisitos de segurança estrutural de uma obra.

<img width="1320" height="679" alt="Preview do Dashboard" src="https://github.com/user-attachments/assets/cdff4cb9-f1d1-4190-98cd-9e781e348286" />

## ⚙️ Tecnologias Utilizadas
* **Ciência de Dados:** Pandas, NumPy, Scikit-Learn, Statsmodels
* **Machine Learning:** XGBoost, SHAP (Explicabilidade)
* **Otimização:** DEAP (Algoritmos Genéticos)
* **Back-end / API:** FastAPI, Uvicorn, Pydantic
* **Front-end / DataViz:** HTML5, Bootstrap 5, Plotly.js, MathJax

---

## 🛡️ Feature Engineering e Blindagem Estatística
Modelos treinados com dados de materiais de construção frequentemente sofrem do chamado **"R² de Vaidade"** devido à alta multicolinearidade (ex: os volumes de areia, brita, água e cimento são matematicamente dependentes para fechar $1m^3$). 

Para garantir que a IA aprendesse a física real e não apenas "decorasse volumes", o projeto passou por uma rigorosa validação:
* **Análise VIF (Variance Inflation Factor):** Identificou-se redundância severa nos agregados inertes.
* **Transformação (Lei de Abrams):** Os inputs brutos de Água e Cimento foram fundidos na variável `water_cement_ratio` (Fator A/C). O modelo foi então blindado, prevendo a resistência exclusivamente com base na pasta química ativa e no tempo de cura, reduzindo o VIF para níveis ideais e estabilizando as predições.

## 🧠 Desempenho do Modelo (XGBoost)
O modelo foi calibrado com dados da *UCI Machine Learning Repository*, unidos a ensaios contemporâneos de misturas sustentáveis. Após a blindagem estatística (804 amostras para treino e 201 para teste), o **XGBoost** provou ser o campeão absoluto da pipeline:

| Modelo Analisado | R² (Validação) | MAE (Erro Médio) | RMSE |
| :--- | :---: | :---: | :---: |
| 🔴 Baseline (Reg. Linear) | 0.5518 | 9.04 MPa | 11.56 MPa |
| 🟡 Random Forest | 0.9150 | 3.53 MPa | 5.03 MPa |
| 🟢 **XGBoost Otimizado** | **0.9229** | **3.01 MPa** | **4.80 MPa** |

*⚙️ **Validação Cruzada:** O modelo final foi ajustado via `RandomizedSearchCV` e submetido a um K-Fold (k=5), atingindo uma média de R² de **0.9224 (+/- 0.0159)**, atestando a sua estabilidade e alta capacidade de generalização para traços inéditos.*

## 📊 Explicabilidade da IA (SHAP)
Para abrir a "caixa preta" do modelo e validar as suas predições face à Engenharia Civil, foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**.

O ranking de importância gerado pelo modelo provou que a Inteligência Artificial mapeou corretamente a fenomenologia do betão/concreto, elegendo as seguintes variáveis como motores da predição:
1. **Idade de Cura (`age_days`):** Fator mais decisivo, refletindo a curva de endurecimento físico.
2. **Fator Água/Cimento (`water_cement_ratio`):** Validando a premissa de que a proporção química é mais crítica que o volume isolado.
3. **Adições Minerais (`slag` e `cement`):** Destacando a escória de alto-forno como um componente essencial para misturas de alta performance sustentáveis.

---
**Desenvolvido por:** Saulo Guimarães Sapucaia - *Data Analyst*
