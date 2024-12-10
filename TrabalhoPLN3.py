import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
from collections import Counter


# Carregar o tokenizador e o modelo DistilBERT pré-treinado
tokenizer = AutoTokenizer.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')
model = AutoModelForSequenceClassification.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')

# Definição das stopwords personalizadas
stopwords_custom = set(['livro', 'pequeno', 'príncipe' , ' " ' , "..." , 'pra' , 'então' , 'sobre'
                        '´´'])

# Função para analisar a opinião de uma avaliação usando o modelo DistilBERT
def analisar_opiniao(avaliacao):
    # Reformatar o texto
    texto_formatado = ' '.join(avaliacao)

    #Reformatar o texto tokenizado
    inputs = tokenizer(texto_formatado, return_tensors="pt", padding=True, truncation=True)

    # Classificar o texto usando o modelo DistilBERT
    outputs = model(**inputs)

    # Obter a probabilidade associada à classe positiva
    negative_probability = torch.softmax(outputs.logits, dim=1)[0][1].item()
    print(f"Probabilidade de ser negativo: {negative_probability}")

    # Ajustar o limiar de decisão para o valor ideal
    threshold = 0.1

    # Decidir o sentimento com base na probabilidade e no limiar
    sentimento = "negativo" if negative_probability >= threshold else "positivo"
    print(f"Sentimento: {sentimento}")

    return sentimento, texto_formatado

# Definição das avaliações
avaliacao1 = '''O Pequeno Príncipe
Esse é um livro que todo mundo já leu então eu tive que ler também! Já vi umas pessoas com tatuagem sobre esse livro e aí não entendo, mas bom, cada um com suas PARTICULARIDADES.
Enfim o livro é bom e bonitinho!''' #elogio

avaliacao2 = '''Li para a escola! Odiei kakaka achei muito chato foi um trauma na minha vida 
kakaka tem muita gente que gosta mais eu não sou uma delas kkkkkk''' #crítica

avaliacao3 = ''' Um mimo
Esse livro é daqueles que te ensinam a ver o mundo nítido. Enxergar a vida com teus próprios olhos,
cada detalhe que te faz gostar dela e cada coisa que não damos a devida desimportância. 
Acredito que seja sim um livro pra crianças, porque só quem ainda tem sua criança interior não destruída pela vida adulta, é capaz de entender a mensagem que o livro quer passar.
Sobre a edição: tradução diferente da original mas ainda assim eu preferi, as ilustrações - um
elemento tão importante pra esse livro - estão impecáveis, e os textos de apoio sempre bem escolhidos, 
amei o do Thiago Queiroz.
Mais um clássico que se justifica em sua importância, leiam.
"Os homens já não têm tempo para conhecer seja lá o que for. Compram coisas prontas no mercado. Mas, como no mercado não há amigos pra vender, os homens já não tem amigos."''' # elogio

avaliacao4 = '''Pequeno príncipe, grande alma...
"A gente corre o risco de chorar um pouco quando se deixou cativar..."
Esta é a segunda vez que leio esse livro, e sinto que é uma obra que precisa ser revisitada de tempos em tempos, porque a cada nova leitura você pode compreender algo novo, ser tocado por algo que sempre esteve ali, mas que ainda não era o momento de você assimilar. É assim que me sinto: muito mais tocada pela sensibilidade do pequeno príncipe hoje, porque desde a última vez que li, já vivi um pouco aqui e ali, então a leitura torna-se certeira em alguns pontos.
Este livro me parece um presente para a criança interior que há em cada adulto. É uma forma de nos lembrar e ensinar sobre a simplicidade da vida, a importância das pequenas coisas, a gratidão e, sobretudo, o amor ? e ouso dizer, o amor maduro. ''' # elogio

avaliacao5 = ''' Ruim
Posso ser julgada por falta de cultura ou sei lá mas pela primeira vez li esse livro com
22 anos e vou dar minha opinião: achei bobo. Me forcei a ler grande parte mas tô me 
sentindo uma idiota em continuar então desisti dele. 
Ok, ele é um clássico mas não me agradou. Vou continuar lendo livro de adulto'''#critica

#Agrupamento
avaliacoes = [avaliacao1, avaliacao2, avaliacao3, avaliacao4, avaliacao5]

# Listas para armazenar palavras positivas e negativas
palavras_positivas = []
palavras_negativas = []

# Pré-processamento e análise de cada avaliação
for i, avaliacao in enumerate(avaliacoes):
    # Tokenização
    tokens = word_tokenize(avaliacao)

    # Remoção de Pontuação
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remoção de Stopwords
    stop_words = set(stopwords.words('portuguese'))
    stop_words.update(stopwords_custom)  # Adicionando as stopwords personalizadas
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Analisar a opinião da avaliação
    opiniao, texto = analisar_opiniao(tokens)

    # Coletar palavras positivas e negativas
    if opiniao == "positivo":
        palavras_positivas.extend(tokens)
        print(f"Avaliação {i+1}: Elogio")
    else:
        palavras_negativas.extend(tokens)
        print(f"Avaliação {i+1}: Crítica")

# Contar a frequência das palavras positivas e negativas
contagem_positivas = Counter(palavras_positivas)
contagem_negativas = Counter(palavras_negativas)

# Gráfico de palavras positivas
if palavras_positivas:
    palavras_positivas, contagens_positivas = zip(*contagem_positivas.most_common(10))
    plt.figure(figsize=(10, 6))
    plt.bar(palavras_positivas, contagens_positivas)
    plt.xlabel('Palavras Positivas')
    plt.ylabel('Frequência')
    plt.title('Palavras Mais Comuns em Avaliações Positivas')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Não há palavras positivas suficientes para criar o gráfico.")

# Gráfico de palavras negativas
if palavras_negativas:
    palavras_negativas, contagens_negativas = zip(*contagem_negativas.most_common(10))
    plt.figure(figsize=(10, 6))
    plt.bar(palavras_negativas, contagens_negativas)
    plt.xlabel('Palavras Negativas')
    plt.ylabel('Frequência')
    plt.title('Palavras Mais Comuns em Avaliações Negativas')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Não há palavras negativas suficientes para criar o gráfico.")
