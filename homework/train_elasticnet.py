#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.wine_quality import print_result, save_estimator, train_best_elasticnet

if __name__ == "__main__":
    result = train_best_elasticnet()
    print_result(result)
    save_estimator(result.estimator)
