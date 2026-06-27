# Classificador de Fases Processuais ⚖️

[![Python](https://img.shields.io/badge/python-3.13.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um projeto educacional de **Machine Learning** para classificação automática de fases processuais judiciais, utilizando técnicas de **Processamento de Linguagem Natural (PLN)** e algoritmos de classificação.

## 📚 Sobre o Projeto

Este classificador foi desenvolvido para categorizar processos judiciais em diferentes fases do PJe (Poder Judiciário Eletrônico) com base no histórico de andamentos e movimentações. O projeto demonstra a aplicação prática de algoritmos de classificação em um contexto real do Direito.

## 🎯 Objetivo Educacional

- Entender conceitos de NLP aplicados a textos jurídicos
- Comparar múltiplos algoritmos de Machine Learning
- Aprender técnicas de validação cruzada e avaliação de modelos
- Visualizar resultados através de matrizes de confusão e boxplots

## 🏛️ Fases Processuais Classificadas

O sistema classifica processos nas seguintes macro-fases, baseado no **Manual PGM-RIO**:

| Código | Fase |
|--------|------|
| 01 | Antes da Sentença |
| 02 | Sentença de 1ª Instância |
| 03 | Recurso 2ª Instância |
| 04 | Tribunais Superiores |
| 05 | Fase de Execução |
| 06 | Suspenso / Sobrestado |
| 07 | Arquivado Definitivamente |

## 🔬 Tecnologias Utilizadas

**Principais dependências (requirements.txt):**
| Biblioteca | Versão |
|------------|--------|
| Python | 3.13.x |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| nltk | 3.9.4 |
| numpy | 2.5.0 |
| matplotlib | 3.11.0 |
| scipy | 1.18.0 |

**Dependência adicional (opcional):**
- `seaborn` - Para heatmaps de confusão mais avançados

## ⚙️ Configuração do Ambiente

### Instalação das Dependências

```bash
# Clone o repositório
git clone https://github.com/phlimasg/classificador-fases-processuais.git
cd "Classificador de Fases Processuais"

# Criar ambiente virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
pip install seaborn  # Opcional: para heatmaps de confusão avançados
```

## 📁 Estrutura do Projeto

```
Classificador de Fases Processuais/
├── ClassificadorTesteInicial.py      # Script inicial com comparação de modelos básicos
├── ClassificadorMultiModels.py       # Avaliação completa com múltiplos algoritmos
├── ClassificadorRandomForest.py      # Versão otimizada com Random Forest
├── dataset1.csv                      # Dataset original
├── dataset2.csv                      # Dataset processado
├── requirements.txt                  # Dependências do projeto
└── README.md
```

## 🤖 Modelos Implementados

### ClassificadorTesteInicial.py
- **KNN** (K-Nearest Neighbors)
- **CART** (Decision Tree)
- **Naive Bayes** (MultinomialNB)
- **SVM** (Support Vector Machine)
- **Random Forest**
- **Logistic Regression**

### ClassificadorMultiModels.py
Todos os modelos acima + **XGBoost** (comentado, disponível para ativação)

## 📊 Pipeline do Projeto

1. **Carregamento dos Dados**: Arquivo CSV com histórico de processos
2. **Preparação de Features**: Combinação de `consulta_rapida`, `lista_autos` e `lista_pav`
3. **Vetorização TF-IDF**: Extração de características com n-grams (1,2)
4. **Treinamento**: Validação cruzada 10-fold
5. **Avaliação**: Classification report e matriz de confusão

## 🖼️ Visualizações Geradas

- **Boxplot**: Comparação de acurácia entre os modelos
- **Matriz de Confusão**: Análise detalhada dos acertos e erros por fase
- **Classification Report**: Métricas por classe (precisão, recall, f1-score)

## 🚀 Como Executar

```bash
# Ativar ambiente virtual (se aplicável)
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Executar um dos scripts
python "ClassificadorTesteInicial.py"
python "ClassificadorMultiModels.py"
python "ClassificadorRandomForest.py"
```

## 🎓 Aprendizado

Este projeto é ideal para:

- Estudantes de Direito que desejam entender automação processual
- Estudantes de TI/Data Science interessados em NLP jurídico
- Pesquisadores que estudam eficiência do Poder Judiciário
- Profissionais que buscam otimizar gestão de processos

## 📝 Licença

Este projeto está licenciado sob MIT - consulte o arquivo `LICENSE` para detalhes.

---

<p align="center">
  <em>Projeto desenvolvido com fins educacionais - Classificação de Fases Processuais Judiciais</em>
</p>

## 👤 Autor

**phlimasg** - Projeto desenvolvido com foco em aplicação prática de Machine Learning no Direito.