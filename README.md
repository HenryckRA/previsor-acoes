# Previsor de Ações com Machine Learning

Projeto de estudo de introdução a Machine Learning com Python: um pipeline completo, da coleta de dados ao gráfico de resultado, que tenta prever o preço de fechamento de uma ação no dia seguinte a partir de médias móveis.

## O que o script faz

1. **Coleta** — baixa o histórico diário da Apple (`AAPL`) de 2019 a 2023 com o `yfinance`.
2. **Engenharia de atributos** — calcula as médias móveis de 7 dias (curto prazo) e 21 dias (médio prazo) do fechamento.
3. **Variável-alvo** — cria `Preco_Amanha` deslocando o fechamento uma linha para cima (`shift(-1)`), e remove as linhas com valores nulos gerados pelas médias e pelo deslocamento.
4. **Treino** — separa 80% dos dias para treino e 20% para teste **sem embaralhar** (`shuffle=False`), respeitando a ordem temporal, e treina uma Regressão Linear.
5. **Avaliação** — calcula o RMSE (erro médio em dólares) e plota preço real × previsão no período de teste.

## Como executar

```bash
git clone https://github.com/HenryckRA/previsor-acoes.git
cd previsor-acoes
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python previsor_acoes.py
```

Para testar outro ativo, altere a variável `ticker` no início do script (ex.: `"PETR4.SA"` para ações da B3).

## Tecnologias

- **yfinance** — coleta dos dados de mercado
- **pandas / numpy** — manipulação dos dados e cálculo dos indicadores
- **scikit-learn** — divisão treino/teste, modelo e métrica
- **matplotlib** — visualização dos resultados

## Limitações e próximos passos

Este é um projeto didático, não uma ferramenta de investimento.

- **Comparação com um baseline ingênuo:** prever o *preço* (e não a variação) faz o RMSE parecer baixo, porque o preço de amanhã é muito próximo do de hoje. O teste justo é comparar o modelo com a previsão "amanhã = hoje".
- **Prever retorno ou direção:** usar como alvo a variação percentual ou se o preço sobe/desce tende a ser um problema mais honesto.
- **Mais atributos e modelos:** volume, RSI, volatilidade e modelos não lineares (Random Forest, gradient boosting).
- **Validação temporal:** usar `TimeSeriesSplit` em vez de um único corte treino/teste.

## Autor

Henrique Rodrigues Alves — Engenharia Mecatrônica, UFU
