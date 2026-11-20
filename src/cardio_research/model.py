"""Preprocessing is refitted within every base learner's out-of-fold training set."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL, NUMERIC

SEED = 42


def preprocessing():
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


def build_model():
    learners = [
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        ("gb", GradientBoostingClassifier(random_state=SEED)),
        ("ada", AdaBoostClassifier(n_estimators=100, random_state=SEED)),
    ]
    return StackingClassifier(
        estimators=[
            (name, Pipeline([("preprocess", preprocessing()), ("model", model)]))
            for name, model in learners
        ],
        final_estimator=LogisticRegression(random_state=SEED, max_iter=1000),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )
