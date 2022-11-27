# Preços de Imóveis em Bangalore, Índia (Modelo de Regressão)
### Projeto de Machine Learning

Neste projeto de ciência de dados, utilizei dados sobre os imóveis disponíveis na cidade de Bangalore na Índia, com o propósito analítico de treinar um modelo de regressão que seja útil para realizar predições sobre o preço dos imóveis de tal cidade indiana. 

Durante o projeto, realizei dois passos sequenciais, que foram **(1)** tratamento de dados e **(2)** construção do modelo de machine learning, na **(1)** etapa do projeto transformei, limpei e manipulei os dados, para que os dados de cada coluna estivéssem adequados para serem implementados no modelo de aprendizagem maquínica, ou seja, nesta fase, removi erros, converti os dados do tipo texto para o tipo numérico, e principalmente lidei com valores atípicos e dados nulos, para que o conjunto de dados estivesse idealmente preparado para ser treinado pelo modelo de regressão que será aplicado.

Já na **(2)** fase construí um modelo de machine learning, isto é, importei o modelo de regressão linear, dividi o conjunto de dados tratado entre dados de treino e dados de teste, e consequentemente treinei o modelo, para depois ter uma análise do quão eficaz era o modelo em realizar previsões sobre o preço dos imóveis, após isto, repeti este mesmo processo repetidas vezes, assim, usei outros modelos de regressão, junto com algumas técnicas como validação cruzada e GridSearch, para saber quais eram os melhores modelos e quais eram os melhores parâmetros dos modelos para realizar previsões mais precisas e acuradas.

### Importação das bibliotecas:

Para este projeto, utilizei Pandas e Numpy para manipulação dos dados, Matplotlib e Seaborn foram usados para visualização de dados, e Sklearn foi a biblioteca útil para a construção do modelo de machine learning.

```
# Importação de bibliotecas:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
```

Após a importação de tais bibliotecas, importei o conjunto de dados e verifique tal dataset contêm 13 mil e 320 linhas, e 9 colunas, abaixo está uma visualização das cinco primeiras linhas do conjunto de dados importado relativo aos imóveis da cidade de Bangalore:

|   | area_type           | availability  | location                 | size      | society | total_sqft | bath | balcony | price  |
|---|---------------------|---------------|--------------------------|-----------|---------|------------|------|---------|--------|
| 0 | Super built-up Area | 19-Dec        | Electronic City Phase II | 2 BHK     | Coomee  | 1056       | 2.0  | 1.0     | 39.07  |
| 1 | Plot Area           | Ready To Move | Chikka Tirupathi         | 4 Bedroom | Theanmp | 2600       | 5.0  | 3.0     | 120.00 |
| 2 | Built-up Area       | Ready To Move | Uttarahalli              | 3 BHK     | NaN     | 1440       | 2.0  | 3.0     | 62.00  |
| 3 | Super built-up Area | Ready To Move | Lingadheeranahalli       | 3 BHK     | Soiewre | 1521       | 3.0  | 1.0     | 95.00  |
| 4 | Super built-up Area | Ready To Move | Kothanur                 | 2 BHK     | NaN     | 1200       | 2.0  | 1.0     | 51.00  |

Nem todas às colunas do dataset foram usadas para prever o preço dos imóveis, além de que antes de construir o modelo de regressão é necessário converter dados textuais em dados numéricos, e principalmente é indispensável lidar com dados ausentes e outliers para não ter erros na construção do modelo preditivo, à partir deste ponto comecei a fase de tratamento e limpeza de dados:

### Tratamento de dados:

Primeiramente, exclui algumas colunas que considerei que não fossem impactantes na previsibilidade de preço dos imóveis, além de que tal exclusão de colunas ajuda na redução de dimensionalidade, e isto ajuda na velocidade de processamento de treino do modelo:

```
df_2 = df.drop(['area_type', 'availability', 'society', 'balcony'],
        axis = 'columns')
```
Após tal exclusão de colunas, obtive esse novo dataframe com menos colunas à serem analisadas:

|   | location                 | size      | total_sqft | bath | price  |
|---|--------------------------|-----------|------------|------|--------|
| 0 | Electronic City Phase II | 2 BHK     | 1056       | 2.0  | 39.07  |
| 1 | Chikka Tirupathi         | 4 Bedroom | 2600       | 5.0  | 120.00 |
| 2 | Uttarahalli              | 3 BHK     | 1440       | 2.0  | 62.00  |
| 3 | Lingadheeranahalli       | 3 BHK     | 1521       | 3.0  | 95.00  |
| 4 | Kothanur                 | 2 BHK     | 1200       | 2.0  | 51.00  |

As colunas do conjunto de dados acima representam informações relativas à:

* location: Localização do imóvel na cidade de Bangalore
* size: Quantidade de quartos presentes no imóvel
* total_sqft: Área total do imóvel em pés quadrados
* bath: Quantidade de banheiros presentes no imóvel
* price: Preço do imóvel 

Após isto, verifiquei a quantidade de dados ausentes no conjunto de dados com o método .isnull().sum() do Pandas:

```
location       1
size          16
total_sqft     0
bath          73
price          0
dtype: int64
```

Acima é observável que há 73 linhas com dados 
