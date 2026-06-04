# Sistema de Classificação de Grãos — Pipeline Clássico

Este projeto implementa o pipeline clássico de visão computacional pedido no trabalho: aquisição, segmentação, extração de features manuais, montagem da matriz X e vetor y, treino de classificadores clássicos e avaliação.

Instruções básicas (PowerShell):

1) Criar e ativar ambiente virtual, instalar dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Extrair features e montar o dataset (os arquivos serão salvos em `outputs/`):

```powershell
python -m notebooks.montar_dataset --archive archive --out outputs/features.csv --max-por-classe 100
```

3) Treinar e avaliar modelos (os artefatos serão salvos em `outputs/`):

```powershell
python -m notebooks.treinar_avaliar --dataset outputs/features.csv --out outputs/resultados --models rf logreg
```

Arquivos importantes:
- `montar_dataset.py`: percorre `archive/`, extrai descritores via `features.py` e gera `dataset.csv`.
- `treinar_avaliar.py`: carrega `dataset.csv`, realiza split estratificado (treino/val/test), normaliza (fit apenas no treino), treina e avalia modelos clássicos, salva métricas e matrizes de confusão.

Sugestões de próximos passos (implementações exigidas pelo trabalho):
- Gerar EDA: boxplots por feature por classe, médias/medianas, comparação visual de distribuições.
- Seleção e análise de features: `SelectKBest`, importância de variáveis por RandomForest, coeficientes da Regressão Logística, PCA para visualização.
- Estudo ablation por grupos de features (cor, textura, forma) e validação cruzada.

Se desejar, prossigo implementando a etapa de EDA, seleção de features e as visualizações acadêmicas em português.
