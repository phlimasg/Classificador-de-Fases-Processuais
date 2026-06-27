import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carga e Limpeza 
dataset = pd.read_csv("dataset2.csv", sep='|', encoding='latin-1')
# =========================================================
# MAPEAMENTO BASEADO NO MANUAL PGM-RIO (FASES_PROCESSUAIS.md)
# =========================================================

mapeamento_fases = {
    # MACRO-FASE 1: 1ª Instância
    'Conhecimento - Antes da Sentença': '01 - Antes da Sentença',
    
    # Agrupamos as sentenças proferidas e as anomalias de "Remessa"
    'Conhecimento - Sentença sem Trânsito em Julgado': '02 - Sentença de 1ª Instância',
    'Conhecimento - Sentença com Trânsito em Julgado': '02 - Sentença de 1ª Instância',
    'Conhecimento - Remessa sem Trânsito em Julgado': '02 - Sentença de 1ª Instância',
    
    # MACRO-FASE 2: 2ª Instância (Agrupamos todos os status de recurso do TJ)
    'Conhecimento - Recurso 2ª Instância - Pendente Julgamento': '03 - Recurso 2ª Instância',
    'Conhecimento - Recurso 2ª Instância - Julgado sem Trânsito': '03 - Recurso 2ª Instância',
    'Conhecimento - Recurso 2ª Instância - Transitado em Julgado': '03 - Recurso 2ª Instância',
    'Conhecimento - Remessa à 2ª Instância - Pendente Julgamento': '03 - Recurso 2ª Instância',
    
    # MACRO-FASE 3: Tribunais Superiores (STJ/STF)
    'Conhecimento - Recurso Tribunais Superiores - Pendente Julgamento': '04 - Tribunais Superiores',
    'Conhecimento - Recurso Tribunais Superiores - Julgado sem Trânsito': '04 - Tribunais Superiores',
    'Conhecimento - Recurso Tribunais Superiores - Transitado em Julgado': '04 - Tribunais Superiores',
    
    # MACRO-FASE 4: Execução (Fases 10, 11, 12, 14 do manual)
    'Execução': '05 - Fase de Execução',
    'Execução Suspensa': '05 - Fase de Execução',
    
    # MACRO-FASE 5: Transversais / Finais
    'Suspenso / Sobrestado': '06 - Suspenso / Sobrestado',
    'Suspensão/Sobrestamento do Processo': '06 - Suspenso / Sobrestado',
    'Arquivado Definitivamente': '07 - Arquivado Definitivamente'
}

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-záéíóúâêôãõç\s]', ' ', texto) 
    return re.sub(r'\s+', ' ', texto).strip()

dataset['texto_feature'] = dataset['consulta_rapida'].apply(limpar_texto) + " " + \
                           dataset['lista_autos'].apply(limpar_texto) + " " + \
                           dataset['lista_pav'].apply(limpar_texto)



# 1. Substitui os 16 nomes soltos por 7 Categorias Oficiais e robustas
dataset['target_agrupado'] = dataset['target'].replace(mapeamento_fases)

contagens = dataset['target_agrupado'].value_counts()
# Identifica quais fases apareceram apenas 1 vez e as remove do dataset
fases_invalidas = contagens[contagens < 2].index
dataset = dataset[~dataset['target_agrupado'].isin(fases_invalidas)]

X = dataset['texto_feature']
y = dataset['target_agrupado']

# 3. VETORIZAÇÃO (Mantendo min_df=3)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3) 
X_vec = vectorizer.fit_transform(X)

# 4. STRATIFY ATIVADO: Agora sem risco de quebrar!
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.25, random_state=7, stratify=y
)

# 4. TREINANDO O MODELO CAMPEÃO
print("Treinando Random Forest Otimizado...")
rf_model = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=7)
# rf_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=7)
rf_model.fit(X_train, y_train)

# ... (pode manter o resto do código de classification_report e matriz de confusão)

# 5. Fazendo Previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# 6. RELATÓRIO DE CLASSIFICAÇÃO (Onde ele erra e onde ele acerta?)
print("\n" + "="*50)
print("RELATÓRIO DE CLASSIFICAÇÃO POR FASE")
print("="*50)
print(classification_report(y_test, y_pred))

# 7. MATRIZ DE CONFUSÃO VISUAL
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)
plt.title('Matriz de Confusão: Onde o modelo está se confundindo?')
plt.ylabel('Fase Real')
plt.xlabel('Fase Prevista pelo Modelo')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()