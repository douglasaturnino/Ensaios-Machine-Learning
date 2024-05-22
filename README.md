# Ensaios de Machine Learning

<div align='center'>
<img src="https://github.com/douglasaturnino/Ensaios-Machine-Learning/assets/95532957/913f0ea6-803c-4299-9df6-00f185322c45"  width=700px/>
</div>

# 1. Contexto
A empresa Data Money fornece serviços de consultoria de Análise e Ciência de Dados para grandes empresas no Brasil e no exterior. O seu principal diferencial de mercado em relação aos concorrentes é o alto retorno financeiro para as empresas clientes, graças a performance de seus algoritmos de Machine Learning.

A Data Money acredita que a expertise no treinamento e ajuste fino dos algoritmos, feito pelos Cientistas de Dados da empresa, é a principal motivo dos ótimos resultados que as consultorias vem entregando aos seus clientes.

Para continuar crescendo a expertise do time, os Cientistas de Dados acreditam que é extremamente importante realizar ensaios nos algoritmos de Machine Learning para adquirir uma experiência cada vez maior sobre o seu funcionamento e em quais cenários as performances são máximas e mínimas, para que a escolha do algoritmo para cada situação seja a mais correta possível.

# 2 Contexto
O contexto deste projeto é realizar ensaios de Machine Learning testando diferentes algoritmos para classificação, regressão e clusterização.

# 3. Estratégia para o Ensaio
A estratégia para este ensaio foi simplesmente criar uma função auxiliar onde todos os algoritmos utilizaram a função para treinar os algoritmos em diferentes situações e com diferentes parâmetros, utilizando conjuntos de dados de treino, validação e teste.

## 3.1 Classificação
Para a classificação, foram utilizados os algoritmos KNN, Decision Tree, Random Forest e Logistic Regression, avaliando as métricas Accuracy, Precision, Recall e F1-Score, sendo a F1-Score a principal métrica, organizada do maior para o menor.

## 3.2 Regressão
Para a regressão, foram utilizados os algoritmos Decision Tree Regressor, Random Forest Regressor, Linear Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polynomial Regression Lasso, Polynomial Regression Ridge e Polynomial Regression Elastic Net, avaliando as métricas R2, MSE, RMSE, MAE e MAPE, sendo o RMSE a principal métrica, organizada do maior para o menor.

## 3.3 Clusterização
Para a clusterização, foram utilizados os algoritmos K-Means e Affinity Propagation, avaliando a métrica silhouette score.

# 4.0 Resultados dos Ensaios

Para os resultados foram retirados os melhores valores de cada algoritmo e formado a tabela abaixo.

## 4.1 Classificação

## Dados de Treino

|           name	        |    accuracy   |   precision	|   recall	 |  f1_score	|                   param                       |
| ------------------------- | ------------- | ------------- | ---------- |------------- | --------------------------------------------- |
|KNeighborsClassifier	    |    0.832186	|   0.812008	|   0.797410 |	0.804643	|   {"n_neighbors": 3}                          |
|DecisionTreeClassifier	    |    0.989644	|   0.992993	|   0.983041 |	0.987992	|   {"max_depth": 19}                           |
|RandomForestClassifier	    |    1.000000	|   1.000000	|   1.000000 |	1.000000	|   {"n_estimators": 300, "max_depth": null}    |
|LogisticRegression	        |    0.876288	|   0.871866	|   0.837661 |	0.854421	|   {"C": 0.1, "solver": "newton-cholesky"}     |

## Dados de Validação

|           name	        |    accuracy   |   precision	|   recall	 |  f1_score	|                   param                       |
| ------------------------- | ------------- | ------------- | ---------- |------------- | --------------------------------------------- |
|KNeighborsClassifier       |	0.676277    |	0.627851    |	0.621278 |	0.624548    |	{"n_neighbors": 3}                          |
|DecisionTreeClassifier     |	0.952444    |	0.956036    |	0.933180 |	0.944469    |	{"max_depth": 14}                           |
|RandomForestClassifier     |	0.965668    |	0.975829    |	0.944168 |	0.959737    |	{"n_estimators": 200, "max_depth": null}    |  
|LogisticRegression         |	0.874320    |	0.869256    |	0.835697 |	0.852146    |	{"C": 0.5, "solver": "newton-cholesky"}     |

## Dados de Teste

|           name	        |    accuracy   |   precision	|   recall	 |  f1_score	|                   param                       |
| ------------------------- | ------------- | ------------- | ---------- |------------- | --------------------------------------------- |
|KNeighborsClassifier	    |0.688449       |	0.648025	|   0.635196 |	0.641546    |	{"n_neighbors": 3}                          |
|DecisionTreeClassifier	    |0.955973       |	0.955781	|   0.943335 |	0.949517    |	{"max_depth": 14}                           |
|RandomForestClassifier	    |0.965937       |	0.974473	|   0.947206 |	0.960646    |	{"n_estimators": 200, "max_depth": null}    |
|LogisticRegression	        |0.871780       |	0.868057	|   0.834756 |	0.851081    |	{"C": 1.0, "solver": "newton-cholesky"}     |

## Conclusões

## KNeighborsClassifier:

Este modelo obteve uma boa acurácia nos dados de treinamento, mas sua performance caiu significativamente nos conjuntos de validação e teste, sugerindo que ele pode não estar generalizando bem para novos dados. Isso pode ser devido à sensibilidade do modelo aos parâmetros escolhidos, como o número de vizinhos.

## DecisionTreeClassifier:

O DecisionTreeClassifier apresentou um desempenho muito bom nos dados de treinamento, mas sua performance foi um pouco inferior nos dados de validação e teste, o que indica que pode haver algum overfitting. A árvore de decisão pode ter sido muito complexa e adaptada demais aos dados de treinamento.

## RandomForestClassifier:

Este modelo teve resultados excelentes em todos os conjuntos de dados, indicando que ele foi capaz de aprender com eficácia os padrões nos dados de treinamento e generalizar bem para novos dados. A floresta aleatória é conhecida por sua capacidade de lidar com overfitting e fornecer boas estimativas mesmo em conjuntos de dados complexos.

## LogisticRegression:

A regressão logística mostrou um desempenho consistente em todos os conjuntos de dados, embora ligeiramente inferior aos resultados do RandomForestClassifier. Isso sugere que a regressão logística pode não ter capturado tão bem a complexidade dos dados quanto a floresta aleatória, mas ainda assim forneceu resultados sólidos e estáveis.

Em geral, enquanto todos os modelos tiveram resultados decentes, o RandomForestClassifier se destacou como o mais robusto, seguido pela regressão logística. Esses resultados destacam a importância de avaliar o desempenho do modelo em conjuntos de dados separados e de escolher métodos de modelagem que possam lidar eficazmente com o overfitting.

## 4.2 Regressão

## Dados de Treino

|              name	         |        R2	    |        MSE 	|    RMSE	 |      MAE	    |    MAPE	|                param                              |
| -------------------------  | --------------   | ------------- | ---------- |------------- | --------- | ------------------------------------------------- |
|LinearRegression            |  0.046058        |	455.996112 	| 21.354065  | 16.998249 	| 8.653186	|    {"fit_intercept": true}                        |
|LinearRegressionLasso       |  0.007401        |	474.474834 	| 21.782443  | 17.305484 	| 8.736697	|    {"alpha": 1, "max_iter": 1500}                 |
|LinearRegressionRidge       |  0.046058        |	455.996401 	| 21.354072  | 16.998308 	| 8.653415	|    {"alpha": 1, "max_iter": 1000}                 |
|LinearRegressionElasticNet  |  0.008744        |	473.833027 	| 21.767706  | 17.290950 	| 8.727685	|    {"alpha": 1, "max_iter": 1500, "l1_ratio": 0.3}|
|DecisionTreeRegressor       |  0.904914        |	45.452203	| 6.741825   | 2.758908  	| 0.548129	|    {"max_depth": 19}                              |
|RandomForestRegressor       |  0.875451        |	59.536012	| 7.715958   | 5.773215  	| 2.816742	|    {"max_depth": 19, "n_estimators": 200}         |
|PolynomialFeatures          |  0.725300        |	131.310015 	| 11.459058  | 7.266166  	| 2.215335	|    {"fit_intercept": true}                        |
|PolynomialFeaturesLasso     |  0.002244        |	476.939911 	| 21.838954  | 17.337843 	| 8.643126	|    {"alpha": 5, "max_iter": 1500}                 |
|PolynomialFeaturesRidge     |  0.262806        |	352.388127 	| 18.772004  | 14.617676 	| 6.916855	|    {"alpha": 5, "max_iter": 1000}                 |
|PolynomialFeaturesElasticNet|  0.006652        |	474.832593 	| 21.790654  | 17.303883 	| 8.703584	|    {"alpha": 2, "max_iter": 1000}                 |

## Dados de Validação

|              name	         |        R2	    |        MSE	|    RMSE	 |      MAE	    |    MAPE	|                param                              |
| -------------------------  | -------------    | ------------- | ---------- |------------- | --------- | ------------------------------------------------- |
|LinearRegression            |  0.039925   	    |  458.447042	| 21.411376  |	17.039754	|  8.682542	|   {"fit_intercept": true}                         |
|Lasso                       |  7.883643e-03	|  473.747081	| 21.765732  |	17.264922	|  8.695808	|   {"alpha": 1, "max_iter": 1500}                  |
|Ridge                       |  0.039933	    |  458.443057	| 21.411283  |	17.038968	|  8.682161	|   {"alpha": 3, "max_iter": 1500}                  |
|ElasticNet                  |  0.008887	    |  473.268144	| 21.754727  |	17.256527	|  8.690803	|   {"alpha": 1, "max_iter": 2000, "l1_ratio": 0.3} |
|DecisionTreeRegressor       |  0.063559	    |  447.161319	| 21.146189  |	16.843452	|  8.395778	|   {"max_depth": 5}                                |
|RandomForestRegressor       |  0.330822	    |  319.540100	| 17.875685  |	13.246470	|  7.155156	|   {"max_depth": 19, "n_estimators": 200}          |
|PolynomialFeatures          |  0.066477	    |  4.457682e+02	| 21.113224  |	16.749939	|  8.547931	|   {"fit_intercept": true}                         |
|PolynomialFeaturesLasso     |  0.002343	    |  476.392774	| 21.826424  |	17.325990	|  8.671844	|   {"alpha": 2, "max_iter": 1500}                  |
|PolynomialFeaturesRidge     |  0.067695	    |  445.186573	| 21.099445  |	16.739734	|  8.575067	|   {"alpha": 2, "max_iter": 1000}                  |
|PolynomialFeaturesElasticNet|  0.006948	    |  474.193795	| 21.775991  |	17.269331	|  8.677584	|   {"alpha": 2, "max_iter": 2000}                  |


## Dados de Teste

|              name	         |        R2	    |        MSE	|    RMSE	 |      MAE	    |    MAPE	|                param                              |
| -------------------------  | -------------    | ------------- | ---------- |------------- | --------- | ------------------------------------------------- |
|LinearRegression            |	0.051166	    | 461.988435	|  21.493916 |	 17.144197	| 8.531355	|   {"fit_intercept": true}                         |
|Lasso                       |	0.007814	    | 483.096411	|  21.979454 |	 17.472410	| 8.752995	|   {"alpha": 1, "max_iter": 2000}                  |
|Ridge                       |	0.051167	    | 461.987749	|  21.493900 |	 17.143729	| 8.532726	|   {"alpha": 2, "max_iter": 1000}                  |
|ElasticNet                  |	0.008836	    | 482.598710	|  21.968129 |	 17.462886	| 8.740648	|   {"alpha": 1, "max_iter": 1000, "l1_ratio": 0.3} |
|DecisionTreeRegressor       |	0.099494	    | 438.457181	|  20.939369 |	 16.699697	| 7.736272	|   {"max_depth": 6}                                |
|RandomForestRegressor       |	0.394800	    | 294.672399	|  17.166025 |	 12.655269	| 6.392848	|   {"max_depth": 19, "n_estimators": 300}          |
|PolynomialFeatures          | 	9.007934e-02    | 4.430413e+02	|  21.048545 |	 16.720535	| 8.242464	|   {"fit_intercept": true}                         |
|PolynomialFeaturesLasso     |	0.002095	    | 485.880941	|  22.042707 |	 17.529487	| 8.720824	|   {"alpha": 2, "max_iter": 2000}                  |
|PolynomialFeaturesRidge     |	0.088422	    | 4.438481e+02	|  21.067702 |	 16.735629	| 8.304551	|   {"alpha": 2, "max_iter": 2000}                  |
|PolynomialFeaturesElasticNet|	0.005895	    | 484.030912	|  22.000703 |	 17.483889	| 8.740537	|   {"alpha": 2, "max_iter": 1000}                  |

## Conclusões

## Dados de Treinamento:
LinearRegression, LinearRegressionLasso, LinearRegressionRidge, LinearRegressionElasticNet:

Todos esses modelos de regressão linear simples têm um RMSE alto nos dados de treinamento, o que sugere que eles não estão capturando bem a variação nos dados. Há uma chance de subajuste.

## DecisionTreeRegressor:

O DecisionTreeRegressor tem um RMSE muito baixo nos dados de treinamento, o que pode ser um sinal de overfitting. O modelo pode ter aprendido os dados de treinamento muito bem, mas pode não generalizar bem para novos dados.

## RandomForestRegressor:

O RandomForestRegressor mostra um RMSE baixo nos dados de treinamento, indicando que está se ajustando bem aos dados. No entanto, como mencionado anteriormente, a diferença entre os resultados nos dados de treinamento e nos dados de validação/teste sugere a possibilidade de overfitting.

## PolynomialFeatures, PolynomialFeaturesLasso, PolynomialFeaturesRidge, PolynomialFeaturesElasticNet:

Esses modelos com características polinomiais têm um RMSE moderado nos dados de treinamento. Eles parecem capturar melhor a variação nos dados do que os modelos de regressão linear simples, mas também podem estar sofrendo de overfitting.
Dados de Validação e Teste:

## LinearRegression, Lasso, Ridge, ElasticNet:

Os modelos de regressão linear simples e suas variantes mostram um RMSE consistente nos conjuntos de dados de validação e teste, em linha com os resultados nos dados de treinamento. No entanto, esses valores de RMSE são bastante altos, sugerindo que esses modelos podem não estar capturando bem a complexidade dos dados.

## DecisionTreeRegressor, RandomForestRegressor:

Ambos os modelos apresentam um RMSE mais alto nos conjuntos de dados de validação e teste em comparação com os dados de treinamento. Isso sugere que eles podem estar sofrendo de overfitting e não generalizando bem para novos dados.

## PolynomialFeatures, PolynomialFeaturesLasso, PolynomialFeaturesRidge, PolynomialFeaturesElasticNet:

Os modelos com características polinomiais também mostram uma diferença significativa entre os resultados nos dados de treinamento e nos dados de validação/teste em termos de RMSE. Isso indica a possibilidade de overfitting e a necessidade de ajustar os modelos para melhor generalização.

## 4.2 Clusterização

|       name         |silhouette score |	    param       |
| ------------------ | --------------  | ----------------   |
|KMeans	             |   0.213219	   | {"n_clusters": 2}  |
|AffinityPropagation |   0.173583	   | {"preference": -7} |


# 5 Lições Aprendidas e Próximos Passos

### Avaliação Holística dos Modelos:
Uma abordagem abrangente na avaliação de modelos é essencial. Além de métricas populares como precisão e acurácia, é fundamental considerar métricas adicionais que reflitam a capacidade do modelo de generalização, como F1-Score, RMSE e silhouette score.

### Identificação de Overfitting e Underfitting:
Os ensaios destacaram a importância de identificar sinais de overfitting e underfitting nos modelos. É crucial encontrar um equilíbrio para evitar que o modelo se adapte demais aos dados de treinamento ou falhe em capturar sua complexidade.

### Necessidade de Regularização:
Modelos simples de regressão linear apresentaram desempenho inadequado nos dados de treinamento, indicando a necessidade de técnicas de regularização para lidar com a subajuste. Métodos como Lasso, Ridge e ElasticNet podem ser aplicados para melhorar a capacidade de generalização.

### Escolha Adequada de Algoritmos:
A seleção do algoritmo certo é crucial para o sucesso do modelo. Os ensaios destacaram que, embora modelos como RandomForestClassifier tenham se destacado, é essencial considerar a adequação do algoritmo para o problema específico e sua capacidade de lidar com overfitting.

Em resumo, os ensaios de Machine Learning destacam a importância da análise cuidadosa dos resultados, da escolha criteriosa dos algoritmos e da prática constante para o desenvolvimento de modelos mais eficazes e generalizáveis.