# 🏗️ Concreto Inteligente: Otimização de Traços com IA

> Aplicação de Machine Learning (XGBoost) e Algoritmos Genéticos (NSGA-II) para previsão de resistência e otimização de custos na Construção Civil. Projeto desenvolvido no ecossistema de Data Science da FIAP.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)

## 🌐 Acesso Online
O dashboard interativo e a API estão hospedados em nuvem e podem ser acessados diretamente pelo navegador:
👉 **[Clique aqui para acessar o Simulador ao vivo](https://saulosapucaia.github.io/concreto_inteligente/)**

---

## 📌 Sobre o Projeto
O cálculo de traços de concreto costuma depender de métodos empíricos e tabelas estáticas. Este projeto introduz uma abordagem orientada a dados, utilizando um modelo **Extreme Gradient Boosting (XGBoost)** treinado com mais de 1.000 amostras laboratoriais. 

O sistema conta com uma API robusta e um Dashboard interativo capaz de:
1. **Laboratório Virtual:** Prever a resistência (MPa) e o custo de uma mistura específica em milissegundos.
2. **Otimizador Financeiro:** Encontrar a combinação de materiais mais barata possível que atenda aos requisitos de segurança estrutural de uma obra.

<img width="1320" height="679" alt="Preview do Dashboard" src="https://github.com/user-attachments/assets/cdff4cb9-f1d1-4190-98cd-9e781e348286" />

## ⚙️ Tecnologias Utilizadas
* **Ciência de Dados:** Pandas, NumPy, Scikit-Learn
* **Machine Learning:** XGBoost, SHAP (Explicabilidade)
* **Otimização:** DEAP (Algoritmos Genéticos)
* **Back-end / API:** FastAPI, Uvicorn, Pydantic
* **Front-end / DataViz:** HTML5, Bootstrap 5, Plotly.js, MathJax

## 🧠 Por que XGBoost? (Modelagem e Desempenho)
O modelo foi calibrado com dados baseados na clássica *UCI Machine Learning Repository*, unida a ensaios contemporâneos de misturas sustentáveis (totalizando 804 amostras para treino e 201 para teste).

Durante a fase de testes, a **Regressão Linear Múltipla** sofreu de *underfitting* severo, pois não conseguiu mapear a não-linearidade da cura do concreto e as interações químicas complexas entre os agregados e aditivos. O **XGBoost** provou ser o campeão absoluto da pipeline:

| Modelo Analisado | R² (Validação) | MAE (Erro Médio) | RMSE |
| :--- | :---: | :---: | :---: |
| 🔴 Baseline (Reg. Linear) | 0.5848 | 8.82 MPa | 11.12 MPa |
| 🟡 Random Forest | 0.9220 | 3.33 MPa | 4.82 MPa |
| 🟢 **XGBoost Otimizado** | **0.9399** | **2.67 MPa** | **4.23 MPa** |

*⚙️ **Validação Cruzada:** O modelo final foi ajustado via `RandomizedSearchCV` e submetido a um K-Fold (k=5), atingindo uma média de R² de **0.9397 (+/- 0.0128)**, o que atesta sua estabilidade e altíssima capacidade de generalização sem overfitting.*

---
**Desenvolvido por:** Saulo Guimarães Sapucaia - *Data Analyst*
