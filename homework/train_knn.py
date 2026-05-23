#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.wine_quality import print_result, save_estimator, train_best_knn

if __name__ == "__main__":
    result = train_best_knn()
    print_result(result)
    save_estimator(result.estimator)
