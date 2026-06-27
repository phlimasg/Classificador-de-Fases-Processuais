import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Baixando as stopwords
nltk.download('stopwords', quiet=True)
stop_words_pt = stopwords.words('portuguese')

# 1. Carregando o dataset exportado e limpo pelo SQL
dataset = pd.read_csv("dataset2.csv", sep='|', encoding='latin-1')

# 2. Preparação das Features
# O banco de dados já nos entregou texto puro, então só precisamos juntar tudo
dataset['texto_feature'] = dataset['consulta_rapida'].astype(str).fillna('') + " " + \
                           dataset['lista_autos'].astype(str).fillna('') + " " + \
                           dataset['lista_pav'].astype(str).fillna('')

# 3. Preparando Variáveis (X = features, y = alvo)
X = dataset['texto_feature']
y = dataset['target']

# 4. Vetorização MELHORADA
# Adicionamos min_df=5 para que termos muito raros não poluam o treinamento
vectorizer = TfidfVectorizer(max_features=5000, 
                             stop_words=stop_words_pt, 
                             ngram_range=(1, 2),
                             min_df=5) 
X_vec = vectorizer.fit_transform(X)

# 5. Divisão de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.30, random_state=7)

# 6. Configuração da Validação Cruzada
num_particoes = 10
kfold = KFold(n_splits=num_particoes, shuffle=True, random_state=7)

# 7. Modelagem COM TODOS OS SEUS MODELOS
np.random.seed(7)
models = []
results = []
names = []

# Adicionando todos os modelos à lista conforme você solicitou:
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('NB', MultinomialNB()))
models.append(('SVM', SVC(kernel='linear')))
models.append(('RF', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=7)))
models.append(('LogReg', LogisticRegression(max_iter=1000, class_weight='balanced')))
models.append(('SVM_Lin_Bal', SVC(kernel='linear', class_weight='balanced'))) # Alterado o nome para diferenciar do SVM acima

# 8. Avaliando um modelo por vez
print("--- Avaliando os modelos ---")
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s:\t Acurácia Média: %.2f%% \t(Desvio Padrão: %f)" % (name, cv_results.mean() * 100, cv_results.std())
    print(msg)

# 9. Boxplot de comparação dos modelos
fig = plt.figure(figsize=(12, 6))
fig.suptitle('Comparação da Acurácia dos Modelos de NLP')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Acurácia')
plt.show()