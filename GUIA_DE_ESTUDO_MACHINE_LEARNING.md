# Guia de estudo — Machine Learning CIMATEC

Este documento resume os projetos, as implementações e os conceitos principais
do repositório. Ele foi escrito como material de preparação para apresentar o
trabalho e responder perguntas sobre as decisões técnicas.

> **Importante:** este guia descreve o código existente no repositório. Nem
> todos os projetos têm o mesmo nível de produção: alguns são estudos,
> notebooks ou exemplos didáticos, enquanto o projeto Olist possui um fluxo
> de Machine Learning mais completo.

## 1. Visão geral do repositório

O repositório reúne cinco frentes principais:

| Projeto | Tema principal | Tipo de dado |
|---|---|---|
| `Olist` | Machine Learning supervisionado, validação, inferência e explicabilidade | Dados tabulares |
| `deep_learning` | Redes neurais com Keras/TensorFlow | Dados tabulares e exemplos de notebooks |
| `image_classifier` | Classificação de imagens com CNN | Imagens |
| `facial_recognition` | Detecção e identificação facial | Imagens de rostos |
| `trabalho-grupo` | Engenharia de prompts e LLMs | Texto e interações com modelos de linguagem |

O fluxo didático geral é:

1. entender e avaliar os dados;
2. preparar os dados;
3. criar features;
4. treinar modelos;
5. validar e comparar modelos;
6. ajustar a decisão do classificador;
7. realizar inferência;
8. explicar as previsões;
9. explorar Active Learning;
10. estudar redes neurais;
11. aplicar redes neurais a imagens;
12. trabalhar com detecção e reconhecimento facial;
13. experimentar engenharia de prompts e LLMs.

---

## 2. Como apresentar o trabalho em uma frase

Uma boa apresentação geral seria:

> “O repositório documenta minha evolução na disciplina, começando pela
> análise e preparação de dados tabulares da Olist, passando por treinamento,
> validação, inferência, explicabilidade e Active Learning, e depois avançando
> para Deep Learning, classificação de imagens, reconhecimento facial e
> engenharia de prompts com LLMs.”

---

# Parte I — Projeto Olist

## 3. Objetivo do projeto Olist

O projeto está em [`Olist/`](./Olist/). Ele utiliza dados públicos de comércio
eletrônico brasileiro para construir um fluxo de Machine Learning.

Os dados envolvem informações como:

- pedidos;
- clientes;
- itens dos pedidos;
- pagamentos;
- avaliações;
- vendedores;
- datas de compra, aprovação, entrega e previsão de entrega;
- localização dos clientes.

O objetivo não é apenas treinar um modelo. O projeto organiza várias etapas:

```text
dados brutos
    ↓
qualidade dos dados
    ↓
análise exploratória
    ↓
engenharia de features
    ↓
separação dos dados
    ↓
treinamento e seleção do modelo
    ↓
cross-validation
    ↓
ajuste de threshold
    ↓
modelo final salvo
    ↓
inferência
    ↓
explicabilidade com SHAP
```

## 4. Estrutura do projeto Olist

Os arquivos mais importantes são:

- [`Olist/module_Olist/dataset.py`](./Olist/module_Olist/dataset.py):
  carregamento, limpeza e criação do dataset;
- [`Olist/module_Olist/features.py`](./Olist/module_Olist/features.py):
  criação de variáveis para o modelo;
- [`Olist/module_Olist/config.py`](./Olist/module_Olist/config.py):
  configurações e caminhos;
- [`Olist/module_Olist/modeling/pipeline.py`](./Olist/module_Olist/modeling/pipeline.py):
  pipelines de pré-processamento e modelos;
- [`Olist/module_Olist/modeling/split.py`](./Olist/module_Olist/modeling/split.py):
  divisão dos dados;
- [`Olist/module_Olist/modeling/train.py`](./Olist/module_Olist/modeling/train.py):
  treinamento e persistência do modelo;
- [`Olist/module_Olist/modeling/cross_validation.py`](./Olist/module_Olist/modeling/cross_validation.py):
  comparação dos modelos e validação cruzada;
- [`Olist/module_Olist/modeling/evaluate.py`](./Olist/module_Olist/modeling/evaluate.py):
  cálculo e organização das métricas;
- [`Olist/module_Olist/inference.py`](./Olist/module_Olist/inference.py):
  utilização do modelo treinado;
- [`Olist/module_Olist/explain.py`](./Olist/module_Olist/explain.py):
  preparação das features para SHAP;
- [`Olist/module_Olist/modeling/interpret.py`](./Olist/module_Olist/modeling/interpret.py):
  criação do explicador e cálculo dos valores SHAP;
- [`Olist/notebooks/`](./Olist/notebooks/):
  notebooks das análises e demonstrações;
- [`Olist/models/`](./Olist/models/):
  modelo serializado e metadados.

---

## 5. Aula 03/04 — Estrutura inicial do Olist

O commit correspondente é:

```text
chore(olist): estrutura inicial do projeto
```

Essa etapa cria a organização base do projeto, incluindo:

- diretório de código;
- diretório de notebooks;
- diretório de modelos;
- documentação;
- testes;
- configuração do projeto;
- arquivo de dependências;
- estrutura para relatórios e referências.

### Por que essa estrutura é útil?

Separar código, notebooks, modelos e relatórios evita colocar todo o projeto
em um único arquivo. Isso melhora:

- manutenção;
- reprodutibilidade;
- colaboração;
- localização dos artefatos;
- separação entre experimentação e código reutilizável.

### Como explicar ao professor

> “A estrutura inicial separa o código-fonte, os notebooks, os modelos
> persistidos, os relatórios e os testes. Assim, os notebooks documentam as
> análises, enquanto o módulo Python concentra as funções reutilizáveis.”

---

## 6. Aula 04 — Qualidade dos dados

O notebook principal é
[`01_orders_profiling.ipynb`](./Olist/notebooks/01_orders_profiling.ipynb).

Antes de treinar um modelo, é necessário saber se os dados são confiáveis.
Essa etapa investiga:

- quantidade de linhas e colunas;
- tipos das variáveis;
- valores ausentes;
- valores duplicados;
- cardinalidade;
- distribuições;
- valores extremos;
- consistência entre tabelas;
- possíveis problemas de datas;
- variáveis que podem causar vazamento de informação.

### Por que essa etapa vem antes do modelo?

Um modelo pode aprender padrões incorretos se os dados tiverem problemas.
Exemplos:

- uma coluna numérica foi lida como texto;
- valores ausentes foram interpretados como zero;
- datas estão em formatos diferentes;
- uma tabela foi combinada com chaves duplicadas;
- uma informação disponível apenas depois do evento foi usada para prever o
  próprio evento.

### Resposta curta

> “A análise de qualidade reduz o risco de treinar o modelo sobre dados
> inconsistentes. Ela permite entender ausências, tipos, duplicidades e
> distribuições antes da preparação das features.”

---

## 7. Aula 05 — Análise exploratória

Os notebooks são:

- [`02_EDA.ipynb`](./Olist/notebooks/02_EDA.ipynb);
- [`02_EDA_template.ipynb`](./Olist/notebooks/02_EDA_template.ipynb).

EDA significa *Exploratory Data Analysis*, ou análise exploratória dos dados.

Enquanto a etapa de qualidade pergunta “os dados estão tecnicamente
consistentes?”, a EDA pergunta “o que esses dados representam e quais padrões
eles possuem?”.

### Perguntas investigadas na EDA

- Quais estados concentram mais clientes?
- Como os pedidos se distribuem ao longo do tempo?
- Qual é o comportamento dos valores dos pedidos?
- Existem pedidos muito pequenos ou muito grandes?
- Como o prazo de entrega varia?
- Há relação entre localização e atraso?
- Como as variáveis se relacionam?
- Existem classes desbalanceadas?

### Diferença entre qualidade e EDA

| Qualidade | EDA |
|---|---|
| Verifica integridade técnica | Busca padrões e hipóteses |
| Tipos, ausências e duplicidades | Distribuições e relações |
| Preparação para confiar nos dados | Preparação para compreender o problema |

---

## 8. Aula 06 — Engenharia de features

A engenharia de features é feita principalmente em:

- [`dataset.py`](./Olist/module_Olist/dataset.py);
- [`features.py`](./Olist/module_Olist/features.py).

Uma *feature* é uma variável que o modelo utiliza para aprender. Dados brutos
geralmente não estão no formato mais útil para o algoritmo.

### Exemplos de transformação

Datas podem ser transformadas em:

- tempo entre compra e aprovação;
- tempo entre aprovação e envio;
- tempo entre envio e entrega;
- prazo prometido;
- diferença entre entrega real e entrega prevista;
- indicador de atraso.

Dados de itens podem ser agregados por pedido:

- quantidade de itens;
- valor total;
- frete total;
- média de preço;
- média de frete.

Dados do cliente podem acrescentar:

- cidade;
- estado;
- localização;
- características regionais.

### O que é o alvo?

O alvo, ou *target*, é a variável que o modelo tenta prever. No projeto, o
fluxo prepara informações relacionadas à entrega e ao atraso.

### Por que não usar os dados brutos diretamente?

Algoritmos trabalham com números e padrões. Uma data como
`2018-08-20 10:30:00` precisa ser convertida em uma representação que contenha
informação útil para o problema.

### Cuidados importantes

Uma feature não pode usar informação que só estaria disponível depois da
previsão. Isso seria vazamento de dados (*data leakage*).

> “A engenharia de features transforma eventos do negócio em variáveis
> numéricas e semânticas que o modelo consegue utilizar, mantendo cuidado para
> não usar informações do futuro.”

---

## 9. Aula extra — Preparação do dataset

O commit:

```text
feat(olist): criação de função main para preparação de dataset
```

é um commit extra que organiza a preparação dos dados.

Ele não substitui uma aula do barema; é um passo intermediário real que liga a
leitura dos dados à modelagem.

Uma função `main` nesse contexto serve para:

1. carregar os arquivos;
2. chamar as funções de transformação;
3. gerar o dataset final;
4. permitir executar o fluxo de forma repetível.

---

## 10. Aula 07 — Modelo treinado

O treinamento está em
[`modeling/train.py`](./Olist/module_Olist/modeling/train.py).

O código suporta modelos como:

- Gradient Boosting;
- XGBoost;
- LightGBM.

### O que acontece no treinamento?

1. O pipeline é criado.
2. O pré-processamento é conectado ao modelo.
3. O modelo recebe `X_train` e `y_train`.
4. O algoritmo aprende os parâmetros.
5. O pipeline completo é salvo com `joblib`.
6. Metadados são salvos em JSON.

### Por que salvar o pipeline completo?

Salvar apenas o estimador pode causar problemas se o pré-processamento usado
no treinamento não for reproduzido na inferência.

Ao salvar o pipeline completo, ficam juntos:

- transformadores;
- codificadores;
- normalizadores, quando houver;
- modelo final.

Assim, novos dados passam pelo mesmo tratamento utilizado no treinamento.

### Modelo e metadados

O modelo é salvo em
[`Olist/models/best_model.joblib`](./Olist/models/best_model.joblib).

Os metadados registram informações como:

- nome do modelo;
- threshold utilizado;
- métrica de seleção;
- métrica utilizada para escolher o threshold.

---

## 11. Aulas 08/09 — Cross-validation

O código está em
[`cross_validation.py`](./Olist/module_Olist/modeling/cross_validation.py).

### Problema de uma única divisão

Se o modelo for avaliado em apenas uma divisão treino/teste, o resultado pode
depender muito dos exemplos que caíram em cada parte.

### Como funciona o `StratifiedKFold`

O projeto utiliza cinco folds, com embaralhamento e `random_state=42`.

Em cada rodada:

1. quatro partes são utilizadas para treinamento;
2. uma parte é utilizada para validação;
3. o processo se repete até todas as partes serem usadas como validação;
4. as métricas são agregadas.

O `Stratified` mantém uma proporção semelhante das classes em cada fold.

### Métricas utilizadas

- **Accuracy:** proporção total de acertos.
- **Precision:** entre os positivos previstos, quantos eram realmente positivos.
- **Recall:** entre os positivos reais, quantos foram encontrados.
- **F1-score:** média harmônica entre precision e recall.
- **ROC AUC:** capacidade de ordenar positivos acima de negativos em diferentes
  thresholds.
- **PR AUC:** área sob a curva Precision-Recall, especialmente útil quando há
  desbalanceamento.

### Média e desvio padrão

O projeto calcula a média e o desvio padrão das métricas nos folds.

- média: desempenho médio;
- desvio padrão: variação do desempenho entre as divisões.

Um modelo com média alta e desvio padrão baixo tende a ser mais consistente.

---

## 12. Ajuste do threshold

Modelos de classificação frequentemente produzem uma probabilidade, não
diretamente uma classe.

Por exemplo:

```text
0.82 -> classe positiva
0.31 -> classe negativa
```

O threshold padrão costuma ser `0.5`, mas nem sempre é o melhor para o
problema.

No código, o threshold é testado de `0.01` a `0.99`, e o valor que maximiza
o F1-score é escolhido.

### Por que ajustar o threshold?

O custo de um falso positivo pode ser diferente do custo de um falso negativo.
Alterar o threshold permite controlar esse equilíbrio.

- threshold menor: tende a aumentar o recall;
- threshold maior: tende a aumentar a precisão;
- threshold escolhido por F1: busca equilíbrio entre precision e recall.

### Out-of-fold predictions

O projeto utiliza previsões *out-of-fold* para escolher o threshold. Cada
previsão é gerada por um modelo que não treinou com aquele exemplo, reduzindo
o risco de avaliar o threshold sobre previsões excessivamente otimistas.

---

## 13. Aula 10 — Inferência

Inferência é usar o modelo já treinado para fazer previsões em dados novos.

Diferença:

| Treinamento | Inferência |
|---|---|
| Aprende parâmetros | Usa parâmetros aprendidos |
| Recebe exemplos e respostas | Recebe exemplos sem resposta |
| Atualiza o modelo | Não deveria atualizar o modelo |

No projeto, o arquivo
[`inference.py`](./Olist/module_Olist/inference.py) representa essa etapa.

### Fluxo de inferência

1. carregar o pipeline salvo;
2. carregar novos dados;
3. aplicar o mesmo pré-processamento;
4. gerar probabilidades ou classes;
5. aplicar o threshold definido;
6. produzir o resultado.

---

## 14. Aula 11 — SHAP Explainability

Os arquivos principais são:

- [`explain.py`](./Olist/module_Olist/explain.py);
- [`modeling/interpret.py`](./Olist/module_Olist/modeling/interpret.py).

SHAP significa *SHapley Additive exPlanations*. A ideia vem dos valores de
Shapley da teoria dos jogos: cada variável recebe uma contribuição para a
previsão.

### O que o SHAP responde?

> “Quais variáveis contribuíram para esta previsão e em que direção?”

Uma feature pode:

- aumentar a previsão;
- diminuir a previsão;
- ter pouca influência;
- variar sua influência conforme o exemplo.

### Por que transformar as features antes?

O pipeline pode conter transformações como codificação categórica. O SHAP
precisa receber uma representação coerente com o modelo que efetivamente
recebeu os dados.

O código:

1. acessa o pré-processador do pipeline;
2. transforma `X`;
3. converte matrizes esparsas para densas quando necessário;
4. obtém os nomes das features transformadas;
5. cria um `DataFrame` com esses nomes;
6. utiliza `shap.TreeExplainer` para o modelo de árvore.

### Por que explicabilidade é importante?

- aumenta a confiança no modelo;
- ajuda a encontrar problemas nos dados;
- auxilia na comunicação com pessoas não técnicas;
- pode revelar vazamento de dados;
- ajuda a decidir quais features revisar.

---

## 15. Aula 12 — Active Learning com Keras

Os arquivos são:

- [`active_learning.py`](./Olist/module_Olist/active_learning.py);
- [`03_active_learning_keras.py`](./Olist/notebooks/03_active_learning_keras.py).

### Ideia principal

No aprendizado supervisionado tradicional, o conjunto rotulado já está
disponível. No Active Learning, rotular dados pode ser caro. Então o modelo
escolhe quais exemplos seriam mais úteis para serem rotulados.

### Fluxo implementado

1. começa com `X_labeled` e `y_labeled`;
2. mantém um pool `X_pool` com exemplos ainda não incorporados;
3. treina um classificador Keras;
4. gera probabilidades para o pool;
5. calcula a incerteza;
6. seleciona as probabilidades mais próximas de `0.5`;
7. incorpora os respectivos rótulos;
8. remove esses exemplos do pool;
9. avalia na validação;
10. repete por várias rodadas.

### Amostragem por incerteza

A incerteza é calculada como:

```python
abs(probabilidade - 0.5)
```

Quanto menor o resultado, mais incerta é a previsão.

Exemplo:

| Probabilidade | Distância até 0.5 | Interpretação |
|---:|---:|---|
| 0.99 | 0.49 | Muito confiante |
| 0.10 | 0.40 | Relativamente confiante |
| 0.51 | 0.01 | Muito incerta |

### O papel de `y_pool`

No exemplo local, `y_pool` representa os rótulos que seriam fornecidos por um
humano depois que as amostras fossem selecionadas. Em um cenário real, um
anotador analisaria essas imagens ou registros e forneceria os rótulos.

### Cuidados da implementação

O código valida:

- tamanhos compatíveis entre dados e rótulos;
- conjunto rotulado não vazio;
- `query_size` positivo;
- número de rodadas positivo;
- número de épocas positivo;
- orçamento de consultas menor que o pool.

### Como explicar ao professor

> “Em vez de rotular aleatoriamente todo o pool, o modelo prioriza exemplos
> sobre os quais está mais incerto. Assim, cada rodada utiliza o orçamento de
> anotação nos dados potencialmente mais informativos.”

---

# Parte II — Deep Learning

## 16. Projeto `deep_learning`

O projeto está em [`deep_learning/`](./deep_learning/).

Ele introduz redes neurais e o uso de Keras/TensorFlow. A estrutura segue um
modelo organizado de projeto, com código, notebooks, testes e configuração.

### Arquivos relevantes

- [`deep_learning/keras.py`](./deep_learning/keras.py);
- [`deep_learning/notebooks/atividade_tensorflow.ipynb`](./deep_learning/notebooks/atividade_tensorflow.ipynb);
- [`deep_learning/notebooks/keras.ipynb`](./deep_learning/notebooks/keras.ipynb);
- [`deep_learning/notebooks/classificador_imagens.ipynb`](./deep_learning/notebooks/classificador_imagens.ipynb).

### Conceitos fundamentais

- **neurônio:** unidade que combina entradas e pesos;
- **camadas:** grupos de neurônios;
- **ativação:** transformação não linear;
- **loss:** mede o erro;
- **otimizador:** atualiza os pesos;
- **época:** uma passagem pelo conjunto de treinamento;
- **batch:** subconjunto processado por vez;
- **validação:** acompanhamento em dados não usados para ajustar diretamente os
  pesos.

### Keras e TensorFlow

Keras oferece uma API de alto nível para construir redes. TensorFlow executa
as operações numéricas e o treinamento.

O exemplo utiliza um dataset tabular de doenças cardíacas baixado de uma URL,
separa treinamento e validação e prepara o uso de camadas Keras.

### Como explicar uma época

> “Uma época é uma passagem completa pelo conjunto de treinamento. Durante
> cada época, o modelo faz previsões, calcula a loss, calcula os gradientes e
> atualiza os pesos pelo otimizador.”

### Overfitting

Overfitting ocorre quando o modelo aprende muito bem os exemplos de
treinamento, mas generaliza mal para dados novos.

Formas de acompanhar ou reduzir:

- conjunto de validação;
- regularização;
- dropout;
- early stopping;
- mais dados;
- arquitetura mais simples.

---

# Parte III — Classificação de imagens

## 17. Projeto `image_classifier`

O projeto está em [`image_classifier/`](./image_classifier/).

Ele organiza uma aplicação de classificação de imagens com uma CNN.

### Arquivos principais

- [`dataset.py`](./image_classifier/image_classifier/dataset.py):
  carregamento e preparação das imagens;
- [`features.py`](./image_classifier/image_classifier/features.py):
  transformações e features;
- [`modeling/train.py`](./image_classifier/image_classifier/modeling/train.py):
  construção e treinamento da CNN;
- [`modeling/predict.py`](./image_classifier/image_classifier/modeling/predict.py):
  avaliação e predição;
- [`plots.py`](./image_classifier/image_classifier/plots.py):
  gráficos;
- [`config.py`](./image_classifier/image_classifier/config.py):
  parâmetros da aplicação;
- [`main.py`](./image_classifier/main.py):
  ponto de entrada.

## 18. CNN utilizada

A rede possui blocos convolucionais com:

- `Conv2D`;
- `BatchNormalization`;
- `MaxPooling2D`;
- `Dropout`.

Depois dos blocos convolucionais:

1. os mapas de características são achatados com `Flatten`;
2. passam por uma camada densa;
3. passam por dropout;
4. chegam à saída `softmax`.

### O que uma convolução faz?

Um filtro percorre a imagem e aprende padrões locais, como:

- bordas;
- texturas;
- formas;
- combinações de padrões.

Camadas iniciais aprendem padrões mais simples. Camadas posteriores combinam
esses padrões para reconhecer estruturas mais complexas.

### Batch Normalization

Normaliza as ativações intermediárias e pode tornar o treinamento mais
estável.

### Max Pooling

Reduz a dimensão espacial dos mapas, mantendo respostas importantes e
diminuindo o custo computacional.

### Dropout

Desativa aleatoriamente parte dos neurônios durante o treinamento. Isso reduz
a dependência excessiva de caminhos específicos e ajuda a combater
overfitting.

## 19. Treinamento da CNN

O modelo é compilado com:

- otimizador Adam;
- `sparse_categorical_crossentropy`;
- métrica de acurácia.

O `sparse_categorical_crossentropy` é apropriado quando os rótulos são
inteiros, como:

```text
0, 1, 2, ..., 9
```

A saída `softmax` produz uma probabilidade para cada classe.

### Early stopping

O treinamento monitora `val_loss`. Se ela não melhorar por determinado número
de épocas:

1. o treinamento é interrompido;
2. os melhores pesos são restaurados;
3. evita-se continuar treinando depois do ponto de melhor generalização.

## 20. Predição de imagens

O arquivo `predict.py`:

1. avalia o modelo no conjunto de teste;
2. imprime loss e acurácia;
3. gera probabilidades;
4. usa `argmax` para escolher a classe com maior probabilidade.

Se a saída do modelo for:

```text
[0.01, 0.03, 0.80, 0.04, ...]
```

o `argmax` retorna o índice da maior probabilidade, neste caso a classe `2`.

---

# Parte IV — Reconhecimento facial

## 21. Projeto `facial_recognition`

O projeto está em [`facial_recognition/`](./facial_recognition/).

Ele simula um sistema de controle de acesso por reconhecimento facial usando
OpenCV e DeepFace.

O fluxo é:

```text
fotos autorizadas
    ↓
detecção dos rostos
    ↓
embeddings faciais
    ↓
cache/base autorizada
    ↓
imagem de teste
    ↓
detecção e embedding
    ↓
comparação por distância cosseno
    ↓
acesso liberado ou negado
    ↓
imagem anotada e relatório CSV
```

## 22. Embeddings faciais

Um embedding é um vetor numérico que representa características de um rosto.

Em vez de comparar pixels diretamente, o sistema compara vetores gerados por
um modelo facial.

Rostos semelhantes tendem a produzir embeddings próximos. Rostos diferentes
tendem a produzir embeddings mais distantes.

## 23. Componentes do projeto

- [`main.py`](./facial_recognition/main.py):
  ponto de entrada da CLI;
- [`src/face_engine.py`](./facial_recognition/src/face_engine.py):
  detecção e geração de embeddings com DeepFace;
- [`src/database.py`](./facial_recognition/src/database.py):
  carrega os autorizados e mantém cache;
- [`src/matcher.py`](./facial_recognition/src/matcher.py):
  calcula distância e decide o resultado;
- [`src/access_control.py`](./facial_recognition/src/access_control.py):
  orquestra o fluxo;
- [`src/image_io.py`](./facial_recognition/src/image_io.py):
  leitura e escrita com OpenCV;
- [`src/visualizer.py`](./facial_recognition/src/visualizer.py):
  caixas e textos nas imagens;
- [`src/config.py`](./facial_recognition/src/config.py):
  modelo, detector, threshold e caminhos.

## 24. Base de autorizados

As fotos em `data/authorized/` representam as pessoas autorizadas.

Na primeira execução:

1. o sistema encontra as imagens;
2. detecta os rostos;
3. calcula os embeddings;
4. associa cada embedding ao nome da imagem ou pasta;
5. pode armazenar o resultado em cache.

O cache evita recalcular os embeddings autorizados em todas as execuções.

## 25. Distância cosseno e decisão

O matcher calcula a distância entre o embedding da imagem de teste e os
embeddings autorizados.

Conceitualmente:

```text
distância pequena  -> maior semelhança
distância grande   -> menor semelhança
```

O threshold define a decisão:

```text
distância <= threshold -> pessoa reconhecida / acesso liberado
distância > threshold  -> acesso negado
```

### Falso positivo e falso negativo

- **Falso positivo:** uma pessoa não autorizada é aceita.
- **Falso negativo:** uma pessoa autorizada é rejeitada.

Em um sistema de controle de acesso, o threshold precisa ser escolhido com
cuidado, porque falsos positivos podem representar um risco de segurança.

## 26. Processamento das imagens

Para cada imagem da pasta de testes, o sistema:

1. tenta ler a imagem;
2. detecta todos os rostos;
3. compara cada rosto com a base autorizada;
4. desenha uma caixa;
5. escreve a decisão;
6. salva a imagem anotada;
7. adiciona o resultado ao CSV.

Se nenhum rosto for detectado, o relatório registra:

```text
sem rosto
```

Isso aconteceu em uma das execuções realizadas no ambiente.

## 27. Pergunta importante: reconhecimento facial é classificação?

É diferente de uma classificação convencional.

Na classificação tradicional, o modelo aprende classes fixas, por exemplo:

```text
gato, cachorro, cavalo
```

No reconhecimento facial, o sistema compara uma representação do rosto com
uma base de identidades. O problema é mais próximo de identificação/verificação
por similaridade.

---

# Parte V — Trabalho em grupo e LLMs

## 28. Projeto `trabalho-grupo`

O projeto está em [`trabalho-grupo/`](./trabalho-grupo/).

Os notebooks são:

- [`engenharia_de_prompt.ipynb`](./trabalho-grupo/engenharia_de_prompt.ipynb);
- [`llm.ipynb`](./trabalho-grupo/llm.ipynb).

O foco é engenharia de prompts e uso de Large Language Models.

## 29. O que é uma LLM?

Uma LLM é um modelo treinado em grandes volumes de texto para prever e gerar
sequências linguísticas.

Ela pode:

- responder perguntas;
- resumir;
- classificar textos;
- extrair informações;
- transformar formatos;
- gerar código;
- seguir instruções.

Ela não funciona exatamente como um banco de dados que consulta uma resposta
pronta. Ela gera uma resposta a partir do contexto e dos padrões aprendidos.

## 30. O que é engenharia de prompt?

É a criação sistemática de instruções para orientar a resposta do modelo.

Um prompt pode especificar:

- papel do modelo;
- objetivo;
- contexto;
- restrições;
- formato de saída;
- exemplos;
- critérios de qualidade.

### Estrutura de um bom prompt

```text
Papel:
  Você é um analista de dados.

Objetivo:
  Explique os resultados de forma clara.

Contexto:
  O público não é técnico.

Restrições:
  Não invente dados e indique incertezas.

Formato:
  Responda com uma tabela e três conclusões.
```

## 31. Zero-shot, few-shot e instruções estruturadas

- **Zero-shot:** o modelo recebe apenas a instrução.
- **Few-shot:** o prompt fornece exemplos do comportamento esperado.
- **Instrução estruturada:** separa contexto, tarefa, restrições e formato.

Exemplos tornam o padrão de resposta mais explícito, mas também podem
introduzir vieses ou fazer o modelo copiar características indesejadas.

## 32. Como avaliar respostas de uma LLM

Uma resposta não deve ser avaliada apenas porque parece bem escrita. É
necessário verificar:

- aderência ao pedido;
- correção factual;
- completude;
- formato;
- clareza;
- consistência;
- presença de informações inventadas;
- tratamento das limitações.

### Alucinação

Alucinação é quando o modelo produz uma informação plausível, mas incorreta
ou sem suporte no contexto.

Para reduzir o risco:

- fornecer contexto confiável;
- pedir que o modelo declare incerteza;
- exigir formato estruturado;
- validar a resposta;
- não tratar a saída como verdade automática.

---

# Parte VI — Perguntas prováveis do professor

## 33. Por que dividir o dataset?

Para medir generalização. O modelo deve ser avaliado em exemplos que não foram
usados diretamente para ajustar seus parâmetros.

## 34. Por que usar cross-validation?

Para reduzir a dependência de uma única divisão e estimar a estabilidade do
desempenho.

## 35. Por que usar `StratifiedKFold`?

Para preservar aproximadamente a proporção das classes em cada fold, algo
importante quando existe desbalanceamento.

## 36. Por que não usar apenas acurácia?

Em classes desbalanceadas, um modelo pode ter acurácia alta e ainda falhar na
classe importante. Precision, recall, F1, ROC AUC e PR AUC fornecem uma visão
mais completa.

## 37. O que é threshold?

É o limite aplicado à probabilidade para transformar uma saída contínua em
classe. O projeto procura o threshold que maximiza F1 nas previsões
out-of-fold.

## 38. O que é data leakage?

É quando uma informação que não estaria disponível no momento da previsão
entra no treinamento. Isso gera uma avaliação artificialmente otimista.

## 39. Por que salvar metadados?

Para registrar como o modelo foi selecionado e qual threshold deve ser usado
na inferência.

## 40. O que o SHAP acrescenta?

Ele mostra a contribuição das features para as previsões, tornando o modelo
mais interpretável.

## 41. O que é Active Learning?

É uma estratégia em que o modelo seleciona os exemplos mais informativos para
serem rotulados, em vez de depender de um conjunto completamente rotulado
desde o início.

## 42. Por que selecionar probabilidades próximas de 0.5?

Porque elas representam maior incerteza no classificador binário. Esses
exemplos podem ajudar o modelo a aprender melhor a fronteira de decisão.

## 43. Qual a diferença entre Machine Learning e Deep Learning?

Machine Learning tradicional costuma depender mais de features projetadas
explicitamente. Deep Learning aprende representações por meio de redes neurais
com várias camadas, especialmente útil para imagens, áudio e texto.

## 44. Para que serve uma CNN?

Para aprender padrões espaciais em imagens por meio de filtros convolucionais.

## 45. Por que usar `softmax` na saída?

Para transformar os logits em probabilidades relativas entre várias classes.

## 46. Por que usar `sparse_categorical_crossentropy`?

Porque as classes são representadas por inteiros, e não por vetores one-hot.

## 47. O que é early stopping?

É interromper o treinamento quando a métrica de validação deixa de melhorar,
restaurando os melhores pesos.

## 48. Como o reconhecimento facial decide quem é a pessoa?

Ele compara o embedding do rosto de teste com embeddings da base autorizada e
verifica se a distância está abaixo do threshold.

## 49. O que acontece se não houver rosto?

O sistema não faz uma identificação e registra a imagem como `sem rosto`.

## 50. O que é engenharia de prompt?

É projetar instruções, contexto, exemplos e formato para obter respostas mais
úteis, consistentes e verificáveis de uma LLM.

---

# Parte VII — Roteiro de apresentação

## 51. Apresentação de dois minutos

> “Meu repositório começa com o projeto Olist. Primeiro analisei a qualidade
> dos dados e fiz uma análise exploratória. Depois realizei engenharia de
> features, preparei o dataset e treinei modelos como Gradient Boosting, XGBoost
> e LightGBM. Para comparar os modelos, usei validação cruzada estratificada,
> várias métricas e ajuste de threshold pelo F1-score. Depois separei a etapa
> de inferência e utilizei SHAP para explicar as previsões.
>
> Também implementei um exemplo de Active Learning com Keras, em que o modelo
> seleciona amostras incertas para receber novos rótulos. Na sequência,
> estudei Deep Learning com Keras e TensorFlow, construí uma CNN para
> classificação de imagens e implementei um sistema de reconhecimento facial
> com OpenCV, DeepFace, embeddings e distância cosseno. Por fim, adicionei o
> projeto em grupo, com notebooks sobre engenharia de prompts e LLMs.”

## 52. Como demonstrar domínio

Ao explicar qualquer projeto, siga esta ordem:

1. **Problema:** o que o projeto tenta resolver?
2. **Dados:** que tipo de dados entram?
3. **Preparação:** como os dados são transformados?
4. **Modelo:** qual algoritmo ou arquitetura é usado?
5. **Avaliação:** como o desempenho é medido?
6. **Resultado:** que artefato é produzido?
7. **Limitações:** o que ainda pode melhorar?

## 53. Limitações que você pode mencionar com segurança

- A qualidade da previsão depende da qualidade e representatividade dos dados.
- O threshold precisa ser validado de acordo com o custo de falsos positivos e
  falsos negativos.
- SHAP explica contribuições do modelo, mas não prova causalidade.
- Active Learning depende da qualidade dos rótulos fornecidos pelo anotador.
- CNNs precisam de dados representativos e podem sofrer overfitting.
- Reconhecimento facial depende da qualidade das imagens, detector, modelo,
  iluminação e threshold.
- Uma LLM pode gerar respostas plausíveis e incorretas; respostas devem ser
  verificadas.

---

# Parte VIII — Resumo final por aula

| Aula | Entrega | O que demonstrar |
|---|---|---|
| 03/04 | Estrutura inicial Olist | Organização modular e reprodutível |
| 04 | Qualidade dos dados | Ausências, tipos, consistência e profiling |
| 05 | EDA | Padrões, distribuições e hipóteses |
| 06 | Feature engineering | Transformação de dados brutos em variáveis úteis |
| 07 | Modelo treinado | Pipelines, treinamento e persistência |
| 08/09 | Cross-validation | Generalização, folds e métricas |
| 10 | Inferência | Uso do modelo em novos dados |
| 11 | SHAP | Explicação das contribuições das features |
| 12 | Active Learning | Seleção de amostras por incerteza |
| 13 | Estrutura Deep Learning | Organização do projeto de redes neurais |
| 14 | Keras | Treinamento, loss, épocas e validação |
| 16 | Estrutura de VC | Organização do projeto de visão |
| 17/18 | Classificador de imagens | CNN, convolução, pooling e softmax |
| 19 | Reconhecimento facial | Embeddings, distância e threshold |
| 20 | Projeto em grupo | Engenharia de prompts e LLMs |

## 54. Frase final

> “O ponto principal do repositório é mostrar o ciclo completo: entender os
> dados, preparar as entradas, treinar, validar, explicar, inferir e aplicar
> diferentes técnicas de Machine Learning e Deep Learning a problemas tabulares,
> visuais e linguísticos.”

