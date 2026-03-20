# 🏗️ Concreto Inteligente: Otimização de Traços com IA

> Aplicação de Machine Learning (XGBoost) e Algoritmos Genéticos (NSGA-II) para previsão de resistência e otimização de custos na Construção Civil. Projeto desenvolvido no ecossistema de Data Science da FIAP.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)

## 🌐 Acesso Online
O dashboard interativo e a API estão hospedados em nuvem e podem ser acessados diretamente pelo navegador:
👉 **[Clique aqui para acessar o Simulador ao vivo] (LINK_DO_SEU_SITE_AQUI)**

---

## 📌 Sobre o Projeto
O cálculo de traços de concreto costuma depender de métodos empíricos e tabelas estáticas. Este projeto introduz uma abordagem orientada a dados, utilizando um modelo **Extreme Gradient Boosting (XGBoost)** treinado com mais de 1.000 amostras laboratoriais. 

O sistema conta com uma API robusta e um Dashboard interativo capaz de:
1. **Laboratório Virtual:** Prever a resistência (MPa) e o custo de uma mistura específica em milissegundos.
2. **Otimizador Financeiro:** Encontrar a combinação de materiais mais barata possível que atenda aos requisitos de segurança estrutural de uma obra.

<img width="1320" height="679" alt="image" src="https://github.com/user-attachments/assets/cdff4cb9-f1d1-4190-98cd-9e781e348286" />

## ⚙️ Tecnologias Utilizadas
* **Ciência de Dados:** Pandas, NumPy, Scikit-Learn
* **Machine Learning:** XGBoost, SHAP (Explicabilidade)
* **Otimização:** DEAP (Algoritmos Genéticos)
* **Back-end / API:** FastAPI, Uvicorn, Pydantic
* **Front-end / DataViz:** HTML5, Bootstrap 5, Plotly.js, MathJax

📊 Estrutura de Dados
O modelo foi calibrado com Data Augmentation, unindo a clássica base da UCI Machine Learning Repository com ensaios contemporâneos de misturas sustentáveis, mapeando 8 variáveis físicas (Cimento, Água, Agregados, Aditivos, etc.).

Métrica de Validação (R²): 0.939

Erro Absoluto Médio (MAE): 2.67 MPa

Desenvolvido por: Saulo Guimarães Sapucaia - Data Analyst
