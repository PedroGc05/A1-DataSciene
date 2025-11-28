# PEDRO GUILHERME CAMPOS || RGM:38974223

# Classificação de Recorrência de Câncer de Mama com Decision Tree

Este projeto tem como objetivo treinar e avaliar um modelo de classificação utilizando o algoritmo Decision Tree aplicado ao dataset `breast-cancer.csv`. O modelo é capaz de prever se um caso representa recorrência ou não da doença, com base em variáveis clínicas categóricas e numéricas.

## Arquivos do Projeto

├── classificador.py
└── breast-cancer.csv


## Descrição do Dataset

O arquivo `breast-cancer.csv` contém 286 registros com informações sobre pacientes diagnosticadas com câncer de mama.  
As variáveis incluem:

- idade  
- estado menopausal  
- tamanho do tumor  
- número de linfonodos  
- presença de cápsula nos linfonodos  
- grau de malignidade  
- lado do seio afetado  
- quadrante da mama  
- exposição à radiação  
- classe (variável alvo)

A coluna alvo é: Class

Com dois valores possíveis:

- `no-recurrence-events`
- `recurrence-events`

## Pré-processamento

As etapas de preparação dos dados incluíram:

1. Remoção de registros com valores ausentes ("?").
2. Conversão de atributos categóricos em variáveis numéricas utilizando One-Hot Encoding.
3. Separação entre atributos (X) e classe (y).

## Treinamento do Modelo

O modelo escolhido foi: DecisionTreeClassifier()


Os dados foram divididos com a técnica Hold-Out:

- 70% para treinamento  
- 30% para teste  

## Avaliação

O script produz:

- acurácia do modelo
- matriz de confusão numérica
- matriz de confusão gráfica
- comparação entre valores reais e previstos

A acurácia obtida foi aproximadamente: 73.8%

Valor coerente com o tamanho e natureza do dataset, que contém diversas variáveis categóricas e é levemente desbalanceado.

## Classificação de Novas Instâncias

O código inclui um exemplo de nova instância, preparada com os mesmos atributos originais. Após aplicar o mesmo processo de One-Hot Encoding e alinhamento das colunas, o modelo é capaz de gerar uma previsão para um caso inédito.

## Execução

Instale as dependências:

```bash
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn matplotlib
```

Execute o script:
python classificador.py

## Considerações Finais

Este projeto demonstra o fluxo completo de construção e avaliação de um modelo de classificação utilizando Decision Trees. Todos os requisitos da atividade foram cumpridos: treinamento do modelo, avaliação com acurácia e matriz de confusão, além da classificação de novas instâncias.
