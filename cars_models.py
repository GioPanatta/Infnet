import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Carregando os dados originais
    data = pd.read_csv('/work/data.csv') 

    # Carregando os dados após a EDA
    modified_data = pd.read_csv('/work/modified_data.csv') 

    st.title("Dashboard de Visualização de Dados e Modelagem com Regressão Linear para a Geely Auto")

    
    st.markdown("""
    **Problema:**

    A empresa automobilística chinesa Geely Auto deseja entrar no mercado dos EUA, produzindo carros localmente para competir com empresas dos EUA e Europa. Contrataram uma consultoria automobilística para entender os fatores que afetam o preço dos carros nos EUA e querem saber:

    - Quais variáveis afetam significativamente o preço dos carros.
    - Quão bem essas variáveis explicam o preço dos carros.
    A consultoria coletou um grande conjunto de dados sobre diferentes tipos de carros no mercado americano.

    **Objetivo:**

    Precisamos modelar o preço dos carros com as variáveis disponíveis. Isso ajudará a administração a entender como os preços variam com essas variáveis e ajustar o design dos carros e a estratégia de negócios para atingir certos níveis de preço. Além disso, o modelo ajudará a compreender a dinâmica de preços em um novo mercado.
    """)

    st.markdown("Abaixo, encontramos um panorama geral da distribuição das nossas variáveis")

    # Criando gráficos de barras verticais para cada variável categórica
    categorical_vars = data.select_dtypes(include=['object']).columns
    for var in categorical_vars:
        if var == 'CarName':
            plt.figure(figsize=(12, 6))
            top_10_car_names = data['CarName'].value_counts().head(10).index
            sns.countplot(y=var, data=data[data['CarName'].isin(top_10_car_names)], palette="Set3")
            plt.title('Top 10 Marcas Mais Comuns')
            plt.xlabel('Contagem')
        else:
            plt.figure(figsize=(10, 4))
            sns.countplot(x=var, data=data, palette="Set2")
            plt.title(f'Distribuição de {var}')
            plt.ylabel('Contagem')
            plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.clf()

    st.markdown("Após o tratamento de erros de digitação dos nomes das marcas, notamos que não houve alteração nas Top 10 marcas mais comuns")
    

    st.markdown("Comparação das Variáveis 'CarName' e 'corrected_car_names'")
    var1 = 'CarName'
    var2 = 'corrected_car_names'

    # Filtrando as top 10 marcas mais comuns para cada variável
    top_10_var1 = data[var1].value_counts().head(10).index
    top_10_var2 = data[var2].value_counts().head(10).index

    # Filtrando os dados para incluir apenas as top 10 marcas
    data_top_10_var1 = data[data[var1].isin(top_10_var1)]
    data_top_10_var2 = data[data[var2].isin(top_10_var2)]

    plt.figure(figsize=(12, 6))

    # Gráfico de barras para a primeira variável
    plt.subplot(1, 2, 1)
    sns.countplot(x=var1, data=data_top_10_var1, palette="Set3")
    plt.title(f'Top 10 Marcas Mais Comuns - {var1}')
    plt.ylabel('Contagem')
    plt.xlabel(var1)
    plt.xticks(rotation=45)

    # Gráfico de barras para a segunda variável
    plt.subplot(1, 2, 2)
    sns.countplot(x=var2, data=data_top_10_var2, palette="Set3")
    plt.title(f'Top 10 Marcas Mais Comuns - {var2}')
    plt.ylabel('Contagem')
    plt.xlabel(var2)
    plt.xticks(rotation=45)

    st.pyplot(plt)
    plt.clf()


    # Boxplot de 'price'
    st.markdown("Em nossa base de dados encontramos outliers na variável price. Em uma base com uma variedade extensa de modelos de veículos, esses outliers são esperados e inclusive úteis para a nossa modelagem preditiva")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['price'])
    plt.title('Boxplot - Price')
    st.pyplot(plt)
    plt.clf()

    # Gráficos de dispersão
    st.markdown("Com base em nossa análise exploratória de dados, podemos concluir que algumas variáveis são bastante significativas para determinar o preço dos carros. Abaixo podemos observar quais dessas variáveis possuem maior correlação com o nosso objetivo")
    high_corr_variables = modified_data.columns.tolist()

    
    cols = 2  
    rows = (len(high_corr_variables) + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    for i, feature in enumerate(high_corr_variables):
        row = i // cols
        col = i % cols
        axs[row, col].scatter(modified_data[feature], modified_data['price'], alpha=0.5)
        axs[row, col].set_title(f'{feature} vs Price')
        axs[row, col].set_xlabel(feature)
        axs[row, col].set_ylabel('Price')

   
    for ax in axs.flat[len(high_corr_variables):]:
        ax.set_visible(False)

    plt.tight_layout()

    # Exibindo os gráficos no Streamlit
    st.pyplot(fig)
    plt.clf()  # Limpando a figura após a exibição

    st.markdown("""
    Com base nas variáveis mais relevantes para o nosso objetivo, podemos extrair alguns insights:

    wheelbase: A distância entre as rodas dianteira e traseira de um veículo, que teve uma correlação positiva com o preço. Carros com uma base de rodas maior tendem a ter preços mais altos, o que pode estar associado à estabilidade do veículo e ao espaço interno
    
    carlength: O comprimento total do carro, que também teve uma correlação positiva com o preço. Carros mais longos podem ser mais caros, talvez devido ao tamanho maior e ao potencial aumento de espaço e conforto.
    
    carwidth: A largura do carro, que mostrou uma correlação positiva com o preço. Veículos mais largos podem oferecer mais espaço interno e ter um design mais robusto, o que pode ser valorizado no mercado.
    
    curbweight: O peso do carro sem passageiros ou carga. Veículos mais pesados muitas vezes têm mais características e equipamentos de segurança, o que pode aumentar o custo.
    
    enginesize: O tamanho do motor do carro, com uma correlação positiva com o preço. Motores maiores geralmente fornecem mais potência e performance, o que pode aumentar o preço do veículo.
    
    boreratio: A relação entre o diâmetro do cilindro e o curso do pistão no motor, que mostrou uma correlação moderada com o preço. Uma maior proporção do furo pode indicar um motor mais potente e eficiente.
    
    horsepower: A quantidade de potência que o motor produz, com uma forte correlação positiva com o preço. Carros com mais cavalos de potência são muitas vezes mais caros devido à maior performance.
    
    citympg e highwaympg: Estas são medidas da eficiência de combustível do carro na cidade e na estrada, respectivamente. Ambas mostraram uma correlação negativa com o preço, indicando que veículos mais eficientes em termos de combustível tendem a ser mais baratos, o que pode refletir uma tendência de veículos menores e mais econômicos.""")

    st.markdown(""" Conclusão:

    Com as variáveis mais correlacionadas com o nosso objetivo, utilizamos um modelo de regressão linear com o qual obtivemos os seguintes resultados:
    MSE (Mean Squared Error - Erro Quadrático Médio): Cerca de 14.323.594,60.
    RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio): Aproximadamente 3.784,65.
    MAE (Mean Absolute Error - Erro Médio Absoluto): Cerca de 2.690,54.)

    O modelo de regressão linear desenvolvido para prever os preços dos carros mostrou um desempenho razoável, com as métricas indicando que as previsões estão, em média, dentro de uma margem de erro de aproximadamente $3,784 em relação ao preço real. Esses resultados são promissores e demonstram que o modelo é capaz de capturar e quantificar as relações entre as características dos carros e seus preços de mercado, especialmente considerando que se trata de um trabalho acadêmico. No entanto, é importante notar que, em um ambiente de negócios real, uma análise mais aprofundada e refinamentos adicionais podem ser necessários para garantir que o modelo atenda aos padrões e requisitos específicos da indústria automobilística""")



if __name__ == "__main__":
    main()


