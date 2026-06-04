from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def modelos_basicos():
    return {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'logreg': LogisticRegression(max_iter=200),
        'svm': SVC(kernel='rbf', probability=True),
        'rf': RandomForestClassifier(n_estimators=100)
    }
