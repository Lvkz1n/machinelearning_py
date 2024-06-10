import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Caminho absoluto para o arquivo CSV
caminho_absoluto = r"C:\Users\Luiz LVK\Documents\GitHub\bot_py\lobby\fakenews.csv"

# Carregamento do dataset
df = pd.read_csv(caminho_absoluto)
print(df.head())

# Converter o texto para minúsculas
df['text'] = df['text'].str.lower()

# Função para remover stopwords
def remove_stopwords_from_column(data, column_name):
    stop_words = TfidfVectorizer(stop_words='english').get_stop_words()
    
    def stopwords_function(text):
        words = text.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words)

    data['text_without_stopwords'] = data[column_name].apply(stopwords_function)
    return data

# Aplicar a função de remoção de stopwords
df = remove_stopwords_from_column(df, 'text')

# Divisão do conjunto de dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['text_without_stopwords'], df['label'], test_size=0.2, random_state=42)

# Criação do modelo Bag-of-Words
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Criação do modelo TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Treinamento do modelo de Regressão Logística
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_tfidf, y_train)

# Previsão no conjunto de teste
y_pred = logistic_regression_model.predict(X_test_tfidf)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")
