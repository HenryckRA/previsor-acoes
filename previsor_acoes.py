import yfinance as yf
import pandas as pd

# 1. Definição do Ativo
# Vamos usar 'AAPL' (Apple)
ticker = "AAPL"

# 2. Coleta Automatizada
# Baixa o histórico de 5 anos para ter uma base sólida de treino
print(f"Iniciando coleta de dados para: {ticker}...")
dados = yf.download(ticker, start="2019-01-01", end="2024-01-01")

# 3. Visualização Inicial
# O comando .head() mostra as primeiras 5 linhas do DataFrame
print("\nPrimeiras linhas da base de dados:")
print(dados.head())

# 4. Verificação da Estrutura
print("\nInformações sobre as colunas e tipos de dados:")
print(dados.info())

#Próximo passo
print("Calculando indicadores matemáticos...")

# 1. Média Móvel de 7 dias (Curto prazo)
#  Coluna 'Close' (Fechamento), e calcula a média de 7 janelas (rolling) e salva numa nova coluna
dados['MM_7'] = dados['Close'].rolling(window=7).mean()

# 2. Média Móvel de 21 dias (Médio prazo)
dados['MM_21'] = dados['Close'].rolling(window=21).mean()

# 3. Limpeza de Dados Extremamente importante
# Os primeiros 20 dias não terão MM_21, gerando valores nulos (NaN).
# Modelos de Machine Learning travam se virem um NaN. Precisa excluí-los.
dados_limpos = dados.dropna()

print("\nBase de dados após tratamento (com Médias Móveis e sem valores nulos):")
print(dados_limpos[['Close', 'MM_7', 'MM_21']].head(10))

# 4. Criando a variável Alvo (Target)
# A ideia é que a IA olhe para as médias móveis de HOJE e tente adivinhar o preço de AMANHÃ.
# A função shift(-1) pega a coluna de fechamento e "puxa" todos os valores uma linha para cima.
dados_limpos['Preco_Amanha'] = dados_limpos['Close'].shift(-1)

# Ao fazer o shift, o último dia da tabela ficará sem o "preço de amanhã" (afinal, amanhã ainda não existe).
# Isso gera um novo NaN na última linha. Precisamos limpar de novo!
dados_limpos = dados_limpos.dropna()

print("\nBase pronta para a IA (Features + Target):")
print(dados_limpos[['Close', 'MM_7', 'MM_21', 'Preco_Amanha']].tail())

#Passo 3:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
## from sklearn.ensemble import RandomForestRegressor

print("\n--- Iniciando o Treinamento da IA ---")

# 1. Separando Entradas (X) e Saídas (y)
# X é a matriz de "dicas" (Features). y é o vetor gabarito (Target).
X = dados_limpos[['MM_7', 'MM_21']]
y = dados_limpos['Preco_Amanha']

# 2. Divisão de Treino e Teste
# Separando 20% dos dados para testar se a IA ficou inteligente mesmo.
# O parâmetro shuffle=False é OBRIGATÓRIO para séries temporais (ações).
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Quantidade de dias para a IA estudar: {len(X_treino)}")
print(f"Quantidade de dias ocultos para testar a IA depois: {len(X_teste)}")

# 3. Criando o Modelo Matemático
# Testei a "Random Forest" (Floresta Aleatória), que cria centenas de árvores de decisão e tira a média delas.
# Porém, a Regressão Linear desempenhou melhor para este caso específico.
modelo = LinearRegression()
## modelo = RandomForestRegressor(n_estimators=100, random_state=42)

# 4. O Treinamento (A Mágica)
# O comando .fit() é onde a pesada computação matemática acontece. 
print("Treinando o modelo... (Isso pode levar alguns segundos)")
modelo.fit(X_treino, y_treino)

# 5. O Teste Cego
# Dias de teste para ela adivinhar o preço de amanhã.
previsoes = modelo.predict(X_teste)

print("Treinamento e previsões concluídos com sucesso!")

#Passo 4
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

print("\n--- Passo 4: Pós-processamento e Resultados ---")

# 1. Calculando o Erro Matemático (RMSE - Root Mean Squared Error)
# O RMSE mede o desvio padrão dos erros da previsão. 
# Em termos simples: mostra quantos Dólares, em média, a IA está errando para cima ou para baixo.
rmse = np.sqrt(mean_squared_error(y_teste, previsoes))
print(f"Margem de Erro Média (RMSE): ${rmse:.2f}")

# 2. Preparando o Gráfico (Matplotlib)
# Plotar o que realmente aconteceu vs. o que a IA tentou adivinhar
plt.figure(figsize=(12, 6))

# Linha Azul: O preço real (Gabarito)
plt.plot(y_teste.index, y_teste, label='Preço Real (Gabarito)', color='blue', linewidth=2)

# Linha Laranja: A previsão da a Inteligência Artificial
plt.plot(y_teste.index, previsoes, label='Previsão da IA', color='orange', linestyle='dashed', linewidth=2)

# 3. Estilização do Gráfico 
plt.title('Previsão de Ações - IA vs Realidade', fontsize=14, fontweight='bold')
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço de Fechamento', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 4. Exibição
print("Gerando gráfico na tela...")
plt.show()