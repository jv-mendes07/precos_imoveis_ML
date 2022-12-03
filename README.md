# Preços de Imóveis em Bangalore, Índia (Modelo de Regressão)
### Projeto de Machine Learning

Neste projeto de ciência de dados, utilizei dados sobre os imóveis disponíveis na cidade de Bangalore na Índia, com o propósito analítico de treinar um modelo de regressão que seja útil para realizar predições sobre o preço dos imóveis de tal cidade indiana. 

Durante o projeto, realizei dois passos sequenciais, que foram **(1)** tratamento de dados e **(2)** construção do modelo de machine learning, na **(1)** etapa do projeto transformei, limpei e manipulei os dados, para que os dados de cada coluna estivéssem adequados para serem implementados no modelo de aprendizagem maquínica, ou seja, nesta fase, removi erros, converti os dados do tipo texto para o tipo numérico, e principalmente lidei com valores atípicos e dados nulos, para que o conjunto de dados estivesse idealmente preparado para ser treinado pelo modelo de regressão que será aplicado.

Já na **(2)** fase construí um modelo de machine learning, isto é, importei o modelo de regressão linear, dividi o conjunto de dados tratado entre dados de treino e dados de teste, e consequentemente treinei o modelo, para depois ter uma análise do quão eficaz era o modelo em realizar previsões sobre o preço dos imóveis, após isto, repeti este mesmo processo repetidas vezes, assim, usei outros modelos de regressão, junto com algumas técnicas como validação cruzada e GridSearch, para saber quais eram os melhores modelos e quais eram os melhores parâmetros dos modelos para realizar previsões mais precisas e acuradas.

## Importação das bibliotecas:

Para este projeto, utilizei Pandas e Numpy para manipulação dos dados, Matplotlib e Seaborn foram usados para visualização de dados, e Sklearn foi a biblioteca útil para a construção do modelo de machine learning.

![](img/35.png)

Após a importação de tais bibliotecas, importei o conjunto de dados e verifique tal dataset contêm 13 mil e 320 linhas, e 9 colunas, abaixo está uma visualização das cinco primeiras linhas do conjunto de dados importado relativo aos imóveis da cidade de Bangalore:

|   | area_type           | availability  | location                 | size      | society | total_sqft | bath | balcony | price  |
|---|---------------------|---------------|--------------------------|-----------|---------|------------|------|---------|--------|
| 0 | Super built-up Area | 19-Dec        | Electronic City Phase II | 2 BHK     | Coomee  | 1056       | 2.0  | 1.0     | 39.07  |
| 1 | Plot Area           | Ready To Move | Chikka Tirupathi         | 4 Bedroom | Theanmp | 2600       | 5.0  | 3.0     | 120.00 |
| 2 | Built-up Area       | Ready To Move | Uttarahalli              | 3 BHK     | NaN     | 1440       | 2.0  | 3.0     | 62.00  |
| 3 | Super built-up Area | Ready To Move | Lingadheeranahalli       | 3 BHK     | Soiewre | 1521       | 3.0  | 1.0     | 95.00  |
| 4 | Super built-up Area | Ready To Move | Kothanur                 | 2 BHK     | NaN     | 1200       | 2.0  | 1.0     | 51.00  |

Nem todas às colunas do dataset foram usadas para prever o preço dos imóveis, além de que antes de construir o modelo de regressão é necessário converter dados textuais em dados numéricos, e principalmente é indispensável lidar com dados ausentes e outliers para não ter erros na construção do modelo preditivo, à partir deste ponto comecei a fase de tratamento e limpeza de dados:

## Tratamento de dados:

Primeiramente, exclui algumas colunas que considerei que não fossem impactantes na previsibilidade de preço dos imóveis, além de que tal exclusão de colunas ajuda na redução de dimensionalidade, e isto ajuda na velocidade de processamento de treino do modelo:

![](img/34.png)

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

* **Dados Nulos**:

![](img/77.png)

Acima é observável que há 73 linhas com dados ausentes na coluna 'bath' relativo à quantidade de banheiro presentes no imóvel, e há 16 linhas com dados ausentes na coluna 'size' relativo à quantidade de quartos de cada imóvel.

Como são poucos dados nulos, decidi exclui-los para ter um conjunto de dados menor e consequentemente prosseguir com o tratamento, poderia ter usado outras técnicas para lidar com dados nulos sem perder informações adicionais, mas considero que está foi a melhor decisão à ser tomada:

![](img/76.png)

Com o método .dropna() aplicado acima, obtive um conjunto de dados menor com 13 mil e 246 linhas.

Concluído o tratamento de dados nulos, comecei a lidar com às variáveis do dataset que eram do tipo categórica e estavam em formato textual.

* **Conversão de tipo de dados**:

A coluna 'size' representa a quantidade de quartos de cada imóvel, porém o tipo de dados da coluna era do tipo object (texto) e tal coluna continha letras presentes em seus valores.

O método .unique() me trouxe a informação sobre os valores únicos da coluna 'size':

![](img/74.png)

É vísivel que não há como converter o tipo de dados da coluna 'size' de object para int, sem antes excluir os elementos textuais que estão presentes nos dados de tal coluna, então para excluir às letras contidas nos dados e deixar somente os números que representam a quantidade de quartos dos imóveis, construí uma função que iria manter somente os números e iria excluir todas às letras que estavam presentes na coluna 'size':

![](img/73.png)


Além de ter mantido somente os números de cada dado da coluna 'size', aproveitei para converter o tipo de dado da coluna de tipo object (texto) para tipo int (numérico), para que pudesse ter tal coluna preparada para ser implementada no modelo de machine learning.

Apliquei novamente o método .unique() para ver o resultado da função .apply() aplicada na coluna 'size':

![](img/72.png)

Após tal conversão da coluna 'size', fui verificar os valores únicos da coluna 'total_sqtf' com o método .unique(), e vi que tal coluna é do tipo object, e que há dados textuais na coluna que impedem a coluna de ser convertida diretamente de object para int:

![](img/71.png)

O traço '-' em um dos dados da coluna foi um dos impecilhos que tive que lidar para transformar essa coluna do tipo texto para o tipo numérico.

Com isto, construí uma função pythônica com o objetivo de filtras os dados da coluna 'total_sqtf' que seriam impedidos de ser convertidos para o tipo numérico por conterem aquele '-' ou por conterem letras que impedissem tal conversão:

![](img/70.png)

A função acima tenta converter o dado para o tipo float (número decimal), caso a função não consiga, então a função retorna False (0), caso contrário retorna True (1).

Usei tal função construída para filtrar os dados da coluna 'total_sqft' que não poderiam ser convertidas diretamente de object para float:

![](img/69.png)

Vi que há 190 dados de tal coluna que não poderiam ser convertido de object para floa diretamente, verifique algumas poucas linhas abaixo da aplicação de tal filtro:

|     |           location |  size |  total_sqft | bath |   price | bhk |
|-----|-------------------:|------:|------------:|-----:|--------:|----:|
|  30 |          Yelahanka | 4 BHK | 2100 - 2850 |  4.0 | 186.000 |   4 |
| 122 |             Hebbal | 4 BHK | 3067 - 8156 |  4.0 | 477.000 |   4 |
| 137 | 8th Phase JP Nagar | 2 BHK | 1042 - 1105 |  2.0 |  54.005 |   2 |
| 165 |           Sarjapur | 2 BHK | 1145 - 1340 |  2.0 |  43.490 |   2 |
| 188 |           KR Puram | 2 BHK | 1015 - 1540 |  2.0 |  56.800 |   2 |

Para retirar os traços de tais dados e deixar somente um valor numérico, decidi calcular a média entre os dois números do lado de cada traço para poder excluir o traço e manter um valor aproximado em relação à área em pés quadrados destes imóveis.

Para este fim, construí mais uma função pythônica:

![](img/68.png)

A função acima separa cada dado com base no traço '-', e assim verifica se esse dado contêm dois valores (que seriam os dois números separados pelo traço), e se tal dado contêm dois valores, então tal função irá retornar a média entre esses dois valores.

Caso o dado não contenha dois valores, então a função irá converter diretamente o valor para o tipo float (número decimal).

Um exemplo de aplicação da função construída seria:

![](img/67.png)

Na primeira linha de código, a função somente transforma o tipo de dado de texto para número decimal diretamente.

A função acima retorna 2475.0 na segunda linha de código, que é a média entre 2100 e 2850.

Após o teste de tal função, apliquei-a sobre a coluna 'total_sqft' para converter todos os dados da coluna diretamente de object para float, e assim consegui ter mais uma coluna preparada para a implementação do modelo de regressão.

![](img/66.png)

* **Redução de dimensionalidade**:

Nesta frase de tratamento dos dados, vi que precisava converter a coluna 'localization' de texto para número inteiro, e para fazer isto teria que usar variáveis dummy para converter cada nome de localização da coluna 'localization' em uma coluna numérica de 0's e 1's que iria ter 1 para afirmar quando um imóvel estaria presente em tal localização, e 0 para negar que tal imóvel estivesse presente em tal localização.

No entanto, para realizar tal transformação, tive que verificar a quantidade de valores únicos na coluna 'localization', isto é, a quantidade de localizações diferentes em que os imóveis de Bangalore estão localizados.

![](img/65.png)

O método len de Python me trouxe o resultado de que há 1.304 localizações diferentes em Bangalore com imóveis presentes, e neste caso se fosse criado uma variável dummy para cada localização, iria acabar tendo 1.304 colunas adicionais no conjunto de dados, ou seja, teríamos mais peso informacional para a análise e consequentemente o modelo de regressão poderia demorar mais tempo para ser treinado e para realizar previsões sobre o preço dos imóveis.

Para evitar esse número exagerado de colunas adicionais, resolvi saber quais eram as localizações que continham mais imóveis e quais que continham menos imóveis em Bangalore.

Usei o método .groupby de Pandas para saber a quantidade de imóveis presentes por localização em Bangalore:

![](img/64.png)

Abaixo está uma visualização breve da quantidade de imóveis por localização:

![](img/62.png)

É notável que há localizações em Bangalore que contém somente um imóvel presente, após tal notação, decidi saber a quantidade de localizações na cidade indiana que há menos de 10 imóveis presentes:

![](img/60.png)

Como saída, obtive que há 1.052 localizações em Bangalore que há 10 ou menos imóveis presentes.

Com tal informações, filtrei as localizações que contêm 10 ou menos imóveis presentes e os atribuí à uma nova variável:

![](img/61.png)

Com tais localizações atribuídas à uma nova variável, decidi substituir cada nome de localização por 'outro', caso tal localização esteja inserida na lista de localizações com 10 ou menos imóveis presentes, ou seja, apliquei essa função abaixo sobre a coluna 'localization':

![](img/59.png)

Caso os nomes de localização da coluna 'localization' estejam inseridos na lista com 10 ou menos imóveis, então os nomes de tais localizações serão substituídos por 'outro' para que todas essas 1.052 localizações diferentes estejam atribuídas à uma mesma classe em comum.

Assim, após tal transformação, consegui diminuir a quantidade de localizações diferentes de 1.304 para 242 localizações.

Neste caso, ao transformar a coluna 'localization' em variável dummy, terei somente 242 colunas adicionais ao invés das 1.304 que seriam tidas.

* **Tratamento de outliers**:

Nessa última fase do tratamento de dados, decidi usar algumas técnicas e avaliar algumas colunas do dataset para saber se há dados atípicos (outliers) em tais colunas ou não.

De início, adicionei uma coluna no dataset que informasse o preço em rupee (moeda indiana) por pé quadrado de cada imóvel de Bangalores, para que posteriormente pudesse identificar valores extremos e assim exclui-los.

![](img/56.png)

Após a criação de tal coluna, filtrei imóveis no dataset que tivessem quartos com uma área menor de 300 pés quadrados, por considerar que tais imóveis poderiam ter quartos com uma área pequena e incomum em relação ao tamanho convencional de quartos dos de mais imóveis.

![](img/55.png)

Por conseguinte, atribuí à uma nova variável somente imóveis que contêm uma área de 300 ou mais pés quadrados por quarto, e assim exclui os imóveis com menos de 300 pés quadrados por quarto por considerar tais imóveis como imóveis com áreas pequenas ou imóveis com dados extremos e atípicos em relação aos outros imóveis.

Depois disto, criei uma função para excluir todos os dados que têm preços por pés quadrados abaixo de um desvio-padrão para baixo e preços por pés quadrados que estão acima de um desvio-padrão para cima, em outras palavras criei uma função para excluir dados extremos e manter somente 63 % dos dados que estão concentrados e próximos em relação às medidas centrais (média, moda, mediana).

![](img/54.png)

Após a criação e aplicação de tal função para excluir valores extremos, obtive um dataset mais reduzido de 10 mil e 241 linhas.

Em continuação para identificar mais outliers, decidi plotar alguns gráficos de dispersão para identificar imóveis que tivessem 3 quartos e fossem mais baratos do que imóveis de 2 quartos, por considerar que o preço de um imóvel está correlacionado também com sua quantidade de cômodos.

![](img/53.png)

Acima é vísivel que criei uma função para plotar gráficos de imóveis de um determinado local que pudesse ser especificado como parâmetro da função, junto com um gráfico de dispersão que expusesse o preço dos imóveis de tal local que contessem somente 2 e 3 quartos.

Apliquei está função para saber o preço dos imóveis de Rajaji Nagar que contêm 2 ou 3 quartos:

![](./img/gra_1.png)

No gráfico acima, observamos que há alguns imóveis com 2 quartos que são mais caros do que imóveis de 3 quartos.

Apliquei a mesma função para obter um gráfico de dispersão relativo aos imóveis localizados no Hebbal:

![](./img/gra_2.png)

Novamente, é observável que há alguns imóveis de 2 quartos mais caros em comparação à imóveis de 3 quartos.

Após identificar esses dados consideravelmente atípicos, decide excluir os imóveis que contêm 3 quartos e são mais baratos do que imóveis de 2 quartos de todos os locais de Bangalore.

Usei uma função pythônica para a exclusão de tais outliers:

![](img/52.png)

Após aplicar tal função sobre o dataset, obtive um conjunto de dados mais reduzido de 7 mil e 329 linhas.

Concluída a exclusão de outliers, plotei novamente os gráficos para ver o preço dos imóveis de 2 e 3 quartos após a exclusão de tais valores atípicos:

![](./img/gra_3.png)

![](./img/gra_6.png)

Como é observado nos dois gráficos acima, excluí os imóveis de 3 quartos que eram mais baratos do que os imóveis de 2 quartos por considerar tais preços de imóveis como valores incomuns com base na quantidade de cômodos.

Por conseguinte, fiz um histograma para visualizar a frequência de imóveis com base no preço por pés quadrado:

![](./img/gra_4.png)

Neste gráfico acima é observado que majoritariamente há mais imóveis com 5000 rupees por pés quadrados, enquanto em contrapartida há uma diminuição de imóveis conforme o preço por pés quadrados aumenta.

Consequentemente, também verifiquei às quantidades de banheiros nos imóveis de Bangalore, e vi que há imóveis com mais de 10 banheiros:

|      | location       | size   | total_sqft | bath | price | bhk | price_per_sqft |
|------|----------------|--------|------------|------|-------|-----|----------------|
| 5277 | Neeladri Nagar | 10 BHK | 4000.0     | 12.0 | 160.0 | 10  | 4.000.000.000  |
| 8486 | other          | 10 BHK | 12000.0    | 12.0 | 525.0 | 10  | 4.375.000.000  |
| 8575 | other          | 16 BHK | 10000.0    | 16.0 | 550.0 | 16  | 5.500.000.000  |
| 9308 | other          | 11 BHK | 6000.0     | 12.0 | 150.0 | 11  | 2.500.000.000  |
| 9639 | other          | 13 BHK | 5425.0     | 13.0 | 275.0 | 13  | 5.069.124.424  |

Como complemento à tal informação, decidi plotar um histograma com a frequência de banheiros registrados nos imóveis de tal cidade indiana:

![](./img/gra_5.png)

É vísivel que a maioria dos imóveis em Bangalore contêm somente de 2 à 4 banheiros.

Como é observado na tabela anterior, há imóveis com mais quartos do que banheiros, e provavelmente isto seria um erro de dado ou seria um tipo de outlier, por ser mais comum que haja mais quartos do que banheiros nos imóveis.

Então, decidi excluir todos os imóveis que contêm mais banheiros do que quartos registrados, por considera-los como outliers:

![](img/51.png)

Após tal exclusão de outliers, fiquei com um dataset de 7 mil linhas e 7 colunas.

* **Criação de variável dummy:**

Concluída a fase de exclusão de outliers (valores atípicos), decidi excluir a coluna 'size' e 'price_per_sqft' por não serem colunas úteis para a construção do modelo de regressão:

![](img/50.png)

Pós esse passo, obtive um dataset neste formato:

|   | location            | total_sqft | bath | price | bhk |
|---|---------------------|------------|------|-------|-----|
| 0 | 1st Block Jayanagar | 2850.0     | 4.0  | 428.0 | 4   |
| 1 | 1st Block Jayanagar | 1630.0     | 3.0  | 194.0 | 3   |
| 2 | 1st Block Jayanagar | 1875.0     | 2.0  | 235.0 | 3   |

Como a coluna 'location' está no formato textual e modelos de machine learning lidam somente com dados numéricos, então terei que converter a coluna 'location' em uma variável dummy de 0's e 1's para cada localização única que terá 0 quando o imóvel não estiver em tal local e que terá 1 quando o imóvel estiver presente em tal local.

![](img/49.png)

A criação de tal variável dummy gerou um dataset com 242 colunas, que representam todas às localizações únicas que contêm imóveis presentes em Bangalore:

|   | 1st Block   Jayanagar | 1st Phase JP   Nagar | 2nd Phase   Judicial Layout | 2nd Stage   Nagarbhavi | 5th Block Hbr   Layout |
|---|-----------------------|----------------------|-----------------------------|------------------------|------------------------|
| 0 | 1                     | 0                    | 0                           | 0                      | 0                      |
| 1 | 1                     | 0                    | 0                           | 0                      | 0                      |
| 2 | 1                     | 0                    | 0                           | 0                      | 0                      |
| 3 | 1                     | 0                    | 0                           | 0                      | 0                      |
| 4 | 1                     | 0                    | 0                           | 0                      | 0                      |

Acima coloquei somente uma amostra de 5 colunas que representam 5 locais em Bangalore que há imóveis localizáveis.

No próximo passo, concatenei o dataset original com o dataset da variável dummy que contêm todas às novas 242 colunas que serão adicionadas ao dataframe original.

![](img/48.png)

Depois disto, excluí a coluna 'localization' por tal coluna não ser mais útil para a construção do modelo de machine learning.

![](img/47.png)

Por fim, acabei ficando com um dataset de 7 mil linhas e 245 colunas para poder criar, treinar e testar o modelo preditivo de regressão linear.

## Criação do modelo de machine learning:

Nesta fase, comecei com a separação do dataset entre duas variáveis X e y, em que na variável X tinha todas às variáveis independentes e preditoras que iriam auxiliar na previsão do preço dos imóveis, e na variável y tinha somente a variável dependente que iria ser utilizada para previsões, ou seja, a coluna de preço dos imóveis.

![](img/47_1.png)
![](img/46.png)

Após a separação do dataset em duas variáveis, importei a função train_test_split() para separar os dados de treino e os dados de teste de cada variável X e y:

![](img/44.png)

Por conseguinte, somente importei o modelo de regressão linear, treinei o modelo com os dados de treino da variável X e y, e por fim obtive a pontuação do quão bom estava o modelo com base nos dados de teste:

![](img/45.png)

O método .score() trouxe uma pontuação de 0.84, ou seja, o modelo está razoavelmente bom em realizar previsões sobre o preço dos imóveis.

No entanto, usei validação cruzada e GridSearch como técnicas para testar outros modelos e outros parâmetros, para saber se há modelos melhores do que o de regressão linear e também para saber os melhores parâmetros que poderiam ser implementados no treinamento destes modelos.

![](img/43.png)

O código acima realiza uma validação cruzada, isto é, utiliza o modelo de regressão linear para treinar iterativamente o modelo com dados de treino diferentes em cada iteração, e por fim o array acima informa a eficácia preditiva do modelo em cada treinamento.

Em média, com base na validação cruzada feita acima teríamos um modelo com uma eficácia de 0.81, que é menor em comparação a pontuação que conseguímos com o modelo de regressão linear sem qualquer validação cruzada.

Após isto, criei uma função para saber quais são os melhores modelos e quais são os melhores parâmetros dos modelos que poderiam ser implementados para prever com mais precisão o preço dos imóveis:

![](img/41.png)
![](img/42.png)

Com a aplicação da função sobre às variáveis X e y, obtive como resultado:

|   | model             | best_score | best_params                                |
|---|-------------------|------------|--------------------------------------------|
| 0 | linear_regression | 0.818354   | {'normalize': True}                        |
| 1 | lasso             | 0.687429   | {'alpha': 1, 'selection': 'cyclic'}        |
| 2 | decision_tree     | 0.712945   | {'criterion': 'mse', 'splitter':   'best'} |

Com a tabela acima é constatável que regressão linear é o melhor modelo para prever o preço dos imóveis de Bangalore, então após tal teste, utilizei regressão linear para prever o preço de alguns possíveis imóveis.

Construí uma função que recebesse a localização, a área em pés quadrados, a quantidade de banheiros e de quartos como parâmetros para prever o preço de tais imóveis com tais características dadas nos parâmetros:

![](img/40.png)

Construída a função para realizar tais previsões em relação ao preço dos imóveis, decidi testar tal função:

![](img/37.png)

Supondo um imóvel localizado em 1st Phase JP Nagar com uma área de 1000 pés quadrados, com 2 quartos e 2 banheiros, nós teriamos previsivelmente um imóvel com um preço aproximado de 83 rupees.

![](img/38.png)

Supondo um imóvel no mesmo local, porém com 3 quartos e 3 banheiros, nós teríamos um imóvel com um preço estimado em 86 rupees.

![](img/39.png)

Já se tivéssemos um imóvel localizado em Indira Nagar, mesmo com 1000 pés quadrados e 3 quartos e banheiros, teríamos neste caso um imóvel com um preço estimado em 184 rupees, ou seja, aparentemente isto constata que previsivelmente os imóveis em Indira Nagar são mais caros do que em 1st Phase JP Nagar.

Concluído tal projeto, espero que tal projeto possa ter lhe trazido alguma noção sobre o processo de ciência de dados, de como passamos pelo tratamento e limpeza de dados, até o momento em que implementamos e treinamos o modelo de machine learning para realizar previsões.

## FIM
