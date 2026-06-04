import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from features import features_grao


def coletar_caminhos(raiz):
    classes = [d for d in os.listdir(raiz) if os.path.isdir(os.path.join(raiz, d))]
    itens = []
    for cls in classes:
        pasta = os.path.join(raiz, cls)
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            for p in glob.glob(os.path.join(pasta, ext)):
                itens.append((p, cls))
    return itens


def montar_dataset(arquivo_archive, saida_csv, max_por_classe=None):
    itens = coletar_caminhos(arquivo_archive)
    linhas = []
    agrupado = {}
    for caminho, rotulo in itens:
        agrupado.setdefault(rotulo, []).append(caminho)

    for rotulo, caminhos in agrupado.items():
        if max_por_classe is not None:
            caminhos = caminhos[:max_por_classe]
        for p in tqdm(caminhos, desc=f"Extraindo {rotulo}"):
            feats = features_grao(p)
            feats['rotulo'] = rotulo
            feats['caminho'] = p
            linhas.append(feats)

    df = pd.DataFrame(linhas)
    df.to_csv(saida_csv, index=False)
    print(f"Dataset salvo em {saida_csv}, shape={df.shape}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='archive', help='Pasta raiz com subpastas por classe')
    parser.add_argument('--out', default='dataset.csv', help='Arquivo CSV de saída')
    parser.add_argument('--max-por-classe', type=int, default=None, help='Máximo de imagens por classe')
    args = parser.parse_args()
    montar_dataset(args.archive, args.out, args.max_por_classe)


if __name__ == '__main__':
    cli()
