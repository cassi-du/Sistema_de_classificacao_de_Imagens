# Sistema de Classificação de Grãos — Pipeline Clássico

Este projeto implementa o pipeline clássico de visão computacional pedido no trabalho: aquisição, segmentação, extração de features manuais, montagem da matriz X e vetor y, treino de classificadores clássicos e avaliação.

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
