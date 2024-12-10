# Análise de Sentimentos com Modelo DistilBERT Multilíngue

Este projeto utiliza o modelo pré-treinado DistilBERT para realizar análises de sentimentos em avaliações textuais. A aplicação realiza a tokenização, análise de sentimentos, remoção de stopwords, e visualiza as palavras mais frequentes em avaliações positivas e negativas.

## Recursos utilizados 

- ``` nltk ```: Tokenização, stopwords, e manipulação de texto.
- ``` transformers ```: Utilização de modelos pré-treinados da Hugging Face.
- ``` torch ```: Para processamento de tensores e utilização do modelo DistilBERT.
- ``` matplotlib ```: Criação de gráficos.
- ``` collections.Counter ``` : Contagem de frequência das palavras

## Funcionalidades

- **Tokenização**: Divide os textos das avaliações em palavras ou tokens.
- **Remoção de Stopwords**: Remove palavras irrelevantes (como artigos, preposições e palavras customizadas).
- **Análise de Sentimentos**:Classifica as avaliações em positivas ou negativas. Utiliza o modelo DistilBERT para gerar probabilidades e calcular o sentimento.
- **Visualização**: Gera gráficos das palavras mais frequentes em avaliações positivas e negativas.

##Como Utilizar

1. **Pré-requisitos**
Certifique-se de ter as seguintes bibliotecas instaladas:

``` nltk ```

``` transformers ```

``` torch ```

``` matplotlib ```

Instale as dependências usando:

` pip install nltk transformers torch matplotlib `

2. **Configuração Inicial** 
Baixe os recursos necessários do nltk:

` import nltk ` 

`  nltk.download('punkt') ` 

` nltk.download('stopwords') ` 

3. **Executando o Código**

Defina suas Avaliações: Insira os textos que deseja analisar na lista avaliacoes.
Personalize Stopwords: Adicione palavras irrelevantes à lista stopwords_custom para melhorar a limpeza do texto.
Execute o script para visualizar:
Sentimento de cada avaliação (positivo ou negativo).
Gráficos de palavras mais frequentes em avaliações positivas e negativas.

## Explicação do Código

1. **Tokenização e Limpeza**
Cada avaliação é dividida em tokens.
Stopwords (incluindo customizadas) e pontuações são removidas.

2. **Análise de Sentimento**
O texto é formatado e tokenizado usando o modelo DistilBERT.
A probabilidade de sentimento negativo é calculada.
Com base em um limiar (definido como 0.1), o sentimento é classificado como:
Negativo: Probabilidade maior ou igual a 0.1.
Positivo: Caso contrário.

3. **Contagem de Palavras**
As palavras das avaliações positivas e negativas são armazenadas separadamente.
Frequências são calculadas utilizando collections.Counter.

4. **Visualização**
Gráficos de barras são gerados para as palavras mais frequentes em cada tipo de avaliação.

## Exemplo de Resultado

Entrada:
Lista de avaliações com sentimentos mistos.

Saída:
Classificação:
Avaliação 1: Positiva

Avaliação 2: Negativa

Avaliação 3: Positiva

Gráficos:
Palavras positivas mais frequentes (exemplo: "amei", "mundo").

Palavras negativas mais frequentes (exemplo: "chato", "trauma").

## Notas
O limiar de decisão pode ser ajustado para melhorar a precisão.
Adicionar mais stopwords pode refinar a análise textual.
Para análises maiores, considere otimizar o uso de memória e tempo de execução.


## Contribuições
Sugestões e melhorias são bem-vindas. Abra uma issue ou envie um pull request!


