import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix

# Import dos Algoritmos de Machine Learning
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# =========================================================
# 1. CONFIGURAÇÕES INICIAIS E CARGA DE DADOS
# =========================================================
nltk.download('stopwords', quiet=True)
stop_words_pt = stopwords.words('portuguese')

# Carregando o dataset exportado e limpo pelo SQL
dataset = pd.read_csv("dataset2.csv", sep='|', encoding='latin-1')

# =========================================================
# 2. MAPEAMENTO DE FASES (MANUAL PGM-RIO)
# =========================================================
mapeamento_fases = {
    # MACRO-FASE 1: 1ª Instância
    'Conhecimento - Antes da Sentença': '01 - Antes da Sentença',
    'Conhecimento - Sentença sem Trânsito em Julgado': '02 - Sentença de 1ª Instância',
    'Conhecimento - Sentença com Trânsito em Julgado': '02 - Sentença de 1ª Instância',
    'Conhecimento - Remessa sem Trânsito em Julgado': '02 - Sentença de 1ª Instância',
    
    # MACRO-FASE 2: 2ª Instância 
    'Conhecimento - Recurso 2ª Instância - Pendente Julgamento': '03 - Recurso 2ª Instância',
    'Conhecimento - Recurso 2ª Instância - Julgado sem Trânsito': '03 - Recurso 2ª Instância',
    'Conhecimento - Recurso 2ª Instância - Transitado em Julgado': '03 - Recurso 2ª Instância',
    'Conhecimento - Remessa à 2ª Instância - Pendente Julgamento': '03 - Recurso 2ª Instância',
    
    # MACRO-FASE 3: Tribunais Superiores
    'Conhecimento - Recurso Tribunais Superiores - Pendente Julgamento': '04 - Tribunais Superiores',
    'Conhecimento - Recurso Tribunais Superiores - Julgado sem Trânsito': '04 - Tribunais Superiores',
    'Conhecimento - Recurso Tribunais Superiores - Transitado em Julgado': '04 - Tribunais Superiores',
    
    # MACRO-FASE 4: Execução
    'Execução': '05 - Fase de Execução',
    'Execução Suspensa': '05 - Fase de Execução',
    
    # MACRO-FASE 5: Transversais / Finais
    'Suspenso / Sobrestado': '06 - Suspenso / Sobrestado',
    'Suspensão/Sobrestamento do Processo': '06 - Suspenso / Sobrestado', 
    'Arquivado Definitivamente': '07 - Arquivado Definitivamente'
}

# Aplicando o agrupamento
dataset['target_agrupado'] = dataset['target'].replace(mapeamento_fases)

# TRAVA DE SEGURANÇA: Remover classes com apenas 1 exemplo
contagens = dataset['target_agrupado'].value_counts()
fases_invalidas = contagens[contagens < 2].index
dataset = dataset[~dataset['target_agrupado'].isin(fases_invalidas)]

# =========================================================
# 3. PREPARAÇÃO DE FEATURES (Unindo as 3 colunas)
# =========================================================

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-záéíóúâêôãõç\s]', ' ', texto) 
    return re.sub(r'\s+', ' ', texto).strip()

dataset['texto_feature'] = dataset['consulta_rapida'].apply(limpar_texto) + " " + \
                           dataset['lista_autos'].apply(limpar_texto) + " " + \
                           dataset['lista_pav'].apply(limpar_texto)

dataset['target_agrupado'] = dataset['target'].replace(mapeamento_fases)
contagens = dataset['target_agrupado'].value_counts()
# Identifica quais fases apareceram apenas 1 vez e as remove do dataset
fases_invalidas = contagens[contagens < 2].index
dataset = dataset[~dataset['target_agrupado'].isin(fases_invalidas)]

X = dataset['texto_feature']
y = dataset['target_agrupado']

# =========================================================
# 4. VETORIZAÇÃO (TF-IDF)
# =========================================================

vectorizer = TfidfVectorizer(max_features=5000, 
                             stop_words=stop_words_pt, 
                             ngram_range=(1, 2),
                             min_df=3) 
X_vec = vectorizer.fit_transform(X)

# =========================================================
# 5. DIVISÃO E VALIDAÇÃO CRUZADA (TESTE DE TODOS OS MODELOS)
# =========================================================
# O stratify=y garante a mesma proporção de fases no treino e no teste
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.25, random_state=7, stratify=y
)

num_particoes = 10
kfold = KFold(n_splits=num_particoes, shuffle=True, random_state=7)

np.random.seed(7)
models = []
results = []
names = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('NB', MultinomialNB()))
models.append(('SVM', SVC(kernel='linear')))
models.append(('RF', RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=7)))
# models.append(('GB', GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=7)))
models.append(('LogReg', LogisticRegression(max_iter=1000, class_weight='balanced')))
models.append(('SVM_Lin_Bal', SVC(kernel='linear', class_weight='balanced')))
# models.append(('XGBoost', XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=7)))

print("--- AVALIANDO TODOS OS MODELOS (Validação Cruzada) ---")
for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s:\t Acurácia Média: %.2f%% \t(Desvio Padrão: %f)" % (name, cv_results.mean() * 100, cv_results.std())
    print(msg)

# =========================================================
# 6. BOXPLOT DE COMPARAÇÃO GERAL
# =========================================================
fig = plt.figure(figsize=(12, 6))
fig.suptitle('Comparação da Acurácia dos Modelos de NLP')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Acurácia')
plt.show()

# =========================================================
# 7. TREINAMENTO FINAL E RAIO-X DO CAMPEÃO (Random Forest)
# =========================================================
print("\n--- TREINANDO O MODELO CAMPEÃO (Random Forest) PARA O RELATÓRIO FINAL ---")
# Usando n_estimators=300 para o modelo final dar o seu máximo
rf_campeao = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=7)
rf_campeao.fit(X_train, y_train)

# Fazendo previsões na base de teste
y_pred = rf_campeao.predict(X_test)

# Relatório de Classificação Detalhado
print("\n" + "="*60)
print("RELATÓRIO DE CLASSIFICAÇÃO POR FASE (MODELO FINAL)")
print("="*60)
print(classification_report(y_test, y_pred))

# Matriz de Confusão Visual
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_campeao.classes_, 
            yticklabels=rf_campeao.classes_)
plt.title('Matriz de Confusão: Random Forest Campeão')
plt.ylabel('Fase Real (Gabarito)')
plt.xlabel('Fase Prevista pelo Modelo')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()