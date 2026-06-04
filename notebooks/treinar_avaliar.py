import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import notebooks.classificacao as modelos_mod
import joblib

SEED = 42


def carregar_dataset(caminho_csv, features_selecionadas_csv=None):
    df = pd.read_csv(caminho_csv)
    df = df.dropna()
    y = df['rotulo'].values if 'rotulo' in df.columns else df['label'].values
    feature_cols = [c for c in df.columns if c not in ('rotulo','label','caminho')]
    if features_selecionadas_csv is not None:
        try:
            sel = pd.read_csv(features_selecionadas_csv)['feature'].tolist()
            # keep only those that exist in dataframe
            feature_cols = [f for f in sel if f in feature_cols]
        except Exception:
            pass
    X = df[feature_cols].values
    nomes_features = feature_cols
    return X, y, nomes_features


def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}


def salvar_matriz_confusao(cm, labels, caminho_saida):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.savefig(caminho_saida)
    plt.close()


def main(csv_dataset, prefixo_saida, modelos_escolhidos=('rf','logreg'), features_selecionadas=None):
    X, y, nomes = carregar_dataset(csv_dataset)
    if features_selecionadas is not None:
        # reload dataset with selected features
        X, y, nomes = carregar_dataset(csv_dataset, features_selecionadas)
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.2, stratify=y_trval, random_state=SEED)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    modelos = modelos_mod.modelos_basicos()
    resultados = {}

    for nome in modelos_escolhidos:
        if nome not in modelos:
            print(f"Modelo {nome} não encontrado; pulando")
            continue
        modelo = modelos[nome]
        grade = {}
        if nome == 'rf':
            grade = {'n_estimators': [50, 100]}
        elif nome == 'logreg':
            grade = {'C': [0.1, 1.0]}

        if grade:
            gs = GridSearchCV(modelo, grade, cv=3, scoring='f1_macro')
            gs.fit(X_train_s, y_train)
            melhor = gs.best_estimator_
        else:
            modelo.fit(X_train_s, y_train)
            melhor = modelo

        melhor.fit(np.vstack([X_train_s, X_val_s]), np.hstack([y_train, y_val]))
        res = avaliar_modelo(melhor, X_test_s, y_test)
        resultados[nome] = res
        joblib.dump({'modelo': melhor, 'scaler': scaler, 'features': nomes}, f"{prefixo_saida}_{nome}.joblib")
        salvar_matriz_confusao(res['confusion_matrix'], labels=np.unique(y), caminho_saida=f"{prefixo_saida}_{nome}_cm.png")

    linhas = []
    for k, v in resultados.items():
        linhas.append({'modelo': k, 'accuracy': v['accuracy'], 'precision': v['precision'], 'recall': v['recall'], 'f1': v['f1']})
    pd.DataFrame(linhas).to_csv(f"{prefixo_saida}_metricas.csv", index=False)
    print('Concluído. Métricas salvas em', f"{prefixo_saida}_metricas.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='outputs/features.csv')
    parser.add_argument('--out', default='outputs/resultados')
    parser.add_argument('--models', nargs='+', default=['rf','logreg'])
    parser.add_argument('--features', default=None, help='CSV com coluna `feature` para selecionar colunas')
    args = parser.parse_args()
    main(args.dataset, args.out, tuple(args.models), features_selecionadas=args.features)
