# Sistema de Classificação de Grãos — Pipeline Clássico

Este projeto implementa o pipeline clássico de visão computacional pedido no trabalho: aquisição, segmentação, extração de features manuais, montagem da matriz X e vetor y, treino de classificadores clássicos e avaliação.
```powershell
projeto/
├── notebooks/
│   ├── 01_segmentacao.ipynb
│   ├── 02_features.ipynb
│   └── 03_classificacao.ipynb
├── outputs/
│   ├── figuras
│   ├── matrizes de confusão
│   ├── tabelas de métricas
│   ├── gráficos de features
│   └── imagens de erros
├── archive/          ← dataset (não incluído no repositório)
├── X.csv
├── y.csv
├── README.md
└── requirements.txt
```
Instruções básicas (PowerShell):

1) Criar e ativar ambiente virtual, instalar dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Baixar dataset e salvar na pasta principal no mesmo nivel que os notebooks e outputs:

Dataset: https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original/code
Extrair pasta e irá estar no local

**3. Executar na ordem dentro da pasta `notebooks/`:**
- `01_segmentacao.ipynb` — segmentação e visualização por classe
- `02_features.ipynb` — extração de features, EDA, seleção e análise
- `03_classificacao.ipynb` — treino, avaliação e comparação dos modelos

## 2.2 Etapas do Pipeline
Etapa 1 — Entrada de imagens: O pipeline inicia com o carregamento das imagens brutas a partir da estrutura de diretórios `archive/`, onde cada subpasta corresponde a uma classe de defeito. Atualmente o carregamento e iteração sobre essas pastas é realizado dentro dos notebooks (principalmente em `02_features.ipynb`), que associam cada imagem ao rótulo de classe a partir do nome da subpasta.

Etapa 2 — Segmentação do grão: A segmentação é implementada na função `segmentar_grao()` presente em `01_segmentacao.ipynb` (e reutilizada em `02_features.ipynb`). A função retorna a imagem BGR original, a versão em escala de cinza e uma máscara binária. O algoritmo combina limiarização automática (método de Otsu) com operações morfológicas (abertura/fechamento) para remover ruído e preencher buracos, e seleciona a componente conectada de maior área como o grão principal.

Etapa 3 — Extração de descritores: Sobre a região segmentada, a função `features_grao()` (em `02_features.ipynb`) extrai um conjunto de 30 descritores numéricos organizados em três categorias: descritores de cor (por exemplo `H_mean`, `S_mean`, `V_mean`, `frac_dark`, histogramas de H), descritores de forma (por exemplo `area`, `circularity`, `solidity`, `eccentricity`, `perimeter`, `extent`, momentos de Hu) e descritores de textura (por exemplo `glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `lbp_var`). Esses descritores capturam aspectos cromáticos, geométricos e texturais úteis para distinguir defeitos.

Etapa 4 — Montagem do dataset tabular: Após a extração, os notebooks consolidam os descritores em `outputs/features.csv`. Cada linha contém os 30 descritores numéricos, o rótulo de classe (`rotulo`) e o caminho da imagem (`caminho`). Este arquivo é a base para as etapas de pré‑processamento e treinamento.

Etapa 5 — Pré-processamento: O notebook `03_classificacao.ipynb` carrega `features.csv` (ou `X.csv`/`y.csv`) e realiza o pré‑processamento: divisão estratificada em treino/val/test (ex.: 64%/16%/20%), e normalização dos atributos com `StandardScaler`. Os parâmetros do scaler são estimados apenas no conjunto de treino e aplicados sem ajuste nos subconjuntos de validação e teste, evitando vazamento de informação.

Etapa 6 — Treino dos classificadores: Os classificadores implementados em `03_classificacao.ipynb` incluem Random Forest (RF), Regressão Logística (LogReg), K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Gaussian Naive Bayes (GNB) e Gradient Boosting (GBM). Para modelos com hiperparâmetros relevantes é usada busca em grade (`GridSearchCV`) com validação cruzada (3 folds) sobre o conjunto de treino, otimizando `f1_macro`. Após seleção, o melhor modelo é treinado na união de treino+validação quando aplicável.

Etapa 7 — Avaliação: A avaliação final é feita sobre o conjunto de teste para obter uma estimativa não enviesada. São calculadas métricas como acurácia, precisão, recall e F1 (versão macro) e geradas matrizes de confusão para análise de erros. Modelos treinados e métricas são salvos em `outputs/` para consulta posterior.

Observação: As funções centrais mantiveram seus nomes (`segmentar_grao()`, `features_grao()`), mas foram migradas para os notebooks; o README e os notebooks referenciam agora essa organização baseada em notebooks em vez de módulos `.py` separados.

## Arquivos gerados em `outputs/`

- `segmentacao_exemplos.png` — exemplos de segmentação por classe
- `features.csv` — tabela completa de features extraídas
- `X.csv` / `y.csv` — matriz de features e vetor de rótulos separados
- `boxplots_features.png` — distribuição das features por classe
- `pca_2d.png` — projeção PCA 2D
- `importancia_rf.png` — importância de variáveis por Random Forest
- `metricas_comparativas.csv` — acurácia, precisão, recall e F1 de todos os modelos
- `matrizes_confusao.png` — matrizes de confusão de todos os modelos
- `imagens_erros.png` — exemplos de erros do melhor modelo
- `comparacao_grupos_features.png` — comparação entre grupos de features
