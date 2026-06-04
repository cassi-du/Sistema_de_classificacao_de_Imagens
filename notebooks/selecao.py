from sklearn.feature_selection import SelectKBest, f_classif

def filter_select(X, y, k=10):
    """Exemplo simples de seleção filter usando ANOVA F-test."""
    sel = SelectKBest(f_classif, k=min(k, X.shape[1]))
    Xs = sel.fit_transform(X, y)
    return Xs, sel
