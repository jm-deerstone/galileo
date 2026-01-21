
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier,
    LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor


ALG_REGISTRY = {
    "random_forest":       RandomForestClassifier,
    "gradient_boosting":   GradientBoostingClassifier,
    "adaboost":            AdaBoostClassifier,
    "extra_trees":         ExtraTreesClassifier,
    "decision_tree":       DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "ridge_classifier":    RidgeClassifier,
    "svm":                 SVC,
    "knn":                 KNeighborsClassifier,
    "gaussian_nb":         GaussianNB,
    "mlp":                 MLPClassifier,

    "random_forest_reg":   RandomForestRegressor,
    "gradient_boosting_reg": GradientBoostingRegressor,
    "adaboost_reg":        AdaBoostRegressor,
    "extra_trees_reg":     ExtraTreesRegressor,
    "decision_tree_reg":   DecisionTreeRegressor,
    "linear_regression":   LinearRegression,
    "ridge":               Ridge,
    "lasso":               Lasso,
    "elastic_net":         ElasticNet,
    "svr":                 SVR,
    "knn_reg":             KNeighborsRegressor,
    "mlp_reg":             MLPRegressor,
}

CLASSIFIERS = {
    "random_forest", "gradient_boosting", "adaboost", "extra_trees", "decision_tree",
    "logistic_regression", "ridge_classifier", "svm", "knn", "gaussian_nb", "mlp",
}
REGRESSORS = {
    "random_forest_reg", "gradient_boosting_reg", "adaboost_reg", "extra_trees_reg", "decision_tree_reg",
    "linear_regression", "ridge", "lasso", "elastic_net", "svr", "knn_reg", "mlp_reg"
}

# 1) per-algorithm HPO search‐spaces
HPO_PARAM_DISTS = {
    "random_forest": {
        "n_estimators": {"type": "int", "range": [50, 500]},
        "max_depth":    {"type": "int_or_none", "range": [5, 50]},
        "max_features": [ "sqrt", "log2"],
    },
    "random_forest_reg": {
        "n_estimators": {"type": "int", "range": [50, 500]},
        "max_depth":    {"type": "int_or_none", "range": [5, 50]},
        "max_features": [ "sqrt", "log2"],
    },
    "gradient_boosting": {
        "n_estimators":  {"type": "int", "range": [50, 200]},
        "learning_rate": {"type": "float", "range": [0.01, 0.2]},
        "max_depth":     {"type": "int", "range": [3, 10]},
    },
    "gradient_boosting_reg": {
        "n_estimators":  {"type": "int", "range": [50, 200]},
        "learning_rate": {"type": "float", "range": [0.01, 0.2]},
        "max_depth":     {"type": "int", "range": [3, 10]},
    },
    "svm": {
        "C":      {"type": "float", "range": [0.1, 100]},
        "kernel": ["rbf", "linear", "poly"],
        "gamma":  ["scale", "auto"],
    },
    "svr": {
        "C":      {"type": "float", "range": [0.1, 100]},
        "kernel": ["rbf", "linear", "poly"],
        "gamma":  ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": {"type": "int", "range": [3, 15]},
        "weights":     ["uniform", "distance"],
    },
    "knn_reg": {
        "n_neighbors": {"type": "int", "range": [3, 15]},
        "weights":     ["uniform", "distance"],
    },
    "logistic_regression": {
        "C":      {"type": "float", "range": [0.01, 100]},
        "solver": ["liblinear", "lbfgs"],
    },
    "ridge": {
        "alpha": {"type": "float", "range": [0.1, 100]},
    },
    "lasso": {
        "alpha": {"type": "float", "range": [0.01, 10]},
    },
    "elastic_net": {
        "alpha":    {"type": "float", "range": [0.01, 10]},
        "l1_ratio": {"type": "float", "range": [0.2, 0.8]},
    },
    "decision_tree": {
        "max_depth":         {"type": "int_or_none", "range": [5, 20]},
        "min_samples_split": {"type": "int", "range": [2, 10]},
    },
    "decision_tree_reg": {
        "max_depth":         {"type": "int_or_none", "range": [5, 20]},
        "min_samples_split": {"type": "int", "range": [2, 10]},
    },
    "mlp": {
        "hidden_layer_sizes": {
            "type": "mlp_hidden_layer_sizes",
            "range": {
                "n_layers": [1, 5],
                "units_per_layer": [10, 200]
            }
        },
        "activation":         ["relu", "tanh"],
        "alpha":              {"type": "float", "range": [0.0001, 0.01]},
    },
    "mlp_reg": {
        "hidden_layer_sizes": {
            "type": "mlp_hidden_layer_sizes",
            "range": {
                "n_layers": [1, 5],
                "units_per_layer": [10, 200]
            }
        },
        "activation":         ["relu", "tanh"],
        "alpha":              {"type": "float", "range": [0.0001, 0.01]},
    },
}