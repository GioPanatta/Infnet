import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

def load_model(file_path):
    return load(file_path)

def main():
    # Carregando os dados
    data = load_data('/work/data.csv')
    modified_data = load_data('/work/modified_data.csv')
    model = load_model('linear_regression_model.joblib')

    st.title("Dashboard de Visualização de Dados e Modelagem com Regressão Linear para a Geely Auto")

    st.markdown("""
    **Problema:**

    A empresa automobilística chinesa Geely Auto deseja entrar no mercado dos EUA, produzindo carros localmente para competir com empresas dos EUA e Europa. Contrataram uma consultoria automobilística para entender os fatores que afetam o preço dos carros nos EUA e querem saber:

    - Quais variáveis afetam significativamente o preço dos carros.
    - Quão bem essas variáveis explicam o preço dos carros. A consultoria coletou um grande conjunto de dados sobre diferentes tipos de carros no mercado americano.

    **Objetivo:**

    Precisamos modelar o preço dos carros com as variáveis disponíveis. Isso ajudará a administração a entender como os preços variam com essas variáveis e ajustar o design dos carros e a estratégia de negócios para atingir certos níveis de preço. Além disso, o modelo ajudará a compreender a dinâmica de preços em um novo mercado.
    """)

    st.markdown("## Análise das Top 10 Marcas Mais Comuns")
    plt.figure(figsize=(12, 6))
    top_10_car_names = data['CarName'].value_counts().head(10).index
    sns.countplot(x='CarName', data=data[data['CarName'].isin(top_10_car_names)], palette="Set3")
    plt.title('Top 10 Marcas Mais Comuns')
    plt.ylabel('Contagem')
    plt.xlabel('Marca')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()
   
    st.markdown("## Histograma da Distribuição dos Preços dos Carros")
    prices = data['price']
    plt.figure(figsize=(10, 6))
    plt.hist(prices, bins=20, color='blue', edgecolor='black')
    plt.title('Histograma da Distribuição dos Preços dos Carros')
    plt.xlabel('Preço')
    plt.ylabel('Frequência')
    st.pyplot(plt)
    plt.clf()

    st.markdown("""
    **Análise de Preços dos Carros:**

    - A distribuição dos preços dos automóveis está concentrada entre a faixa dos $5,000 a $17,500.
    - Valores em Dólares americanos:
        - Média do Preço: $13,276.71
        - Mediana do Preço: $10,295.00
        - Desvio Padrão do Preço: $7,988.85
        - Preço Mínimo: $5,118.00
        - Preço Máximo: $45,400.00
    """)


    st.markdown("## Gráficos de Regressão Linear com as Variáveis Mais Correlacionadas ao Preço dos Automóveis")
    selected_features = [
        'wheelbase', 'carlength', 'carwidth', 'curbweight', 
        'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg'
    ]
    for feature in selected_features:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=feature, y='price', data=data)
        plt.title(f'Regressão Linear - {feature} vs Price')
        st.pyplot(plt)
        plt.clf()

    st.markdown("""
    **Insights baseados nas variáveis que mais influenciam nos Preços dos Carros:**
    
    - **wheelbase**: Carros com base de rodas maior tendem a ter preços mais altos.
    - **carlength**: Carros mais longos podem ser mais caros.
    - **carwidth**: Veículos mais largos podem oferecer mais espaço interno.
    - **curbweight**: Veículos mais pesados tendem a ter mais recursos e equipamentos de segurança.
    - **enginesize**: Motores maiores geralmente oferecem mais potência.
    - **boreratio**: Uma maior proporção do furo pode indicar um motor mais potente.
    - **horsepower**: Carros com mais cavalos de potência são frequentemente mais caros.
    - **citympg** e **highwaympg**: Correlação negativa com o preço, indicando que veículos mais eficientes em termos de combustível tendem a ser mais baratos.
    """)

    st.markdown("""
    **Conclusão:**

    Com as variáveis mais correlacionadas com o nosso objetivo, utilizamos um modelo de regressão linear com o qual obtivemos os seguintes resultados:
    - MSE (Mean Squared Error - Erro Quadrático Médio): Cerca de 14.323.594,60.
    - RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio): Aproximadamente 3.784,65.
    - MAE (Mean Absolute Error - Erro Médio Absoluto): Cerca de 2.690,54.

    O modelo de regressão linear desenvolvido para prever os preços dos carros mostrou um desempenho razoável, com as métricas indicando que as previsões estão, em média, dentro de uma margem de erro de aproximadamente $3,784 em relação ao preço real. Esses resultados são promissores e demonstram que o modelo é capaz de capturar e quantificar as relações entre as características dos carros e seus preços de mercado.
    """)

if __name__ == "__main__":
    main()


