"""Model zoo behind one interface: fit(X, y, X_val, y_val) -> self; predict(X) -> p.
All models see identical features/folds and are decoded + scored identically."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

SEED = config.SEED
NJ = config.N_JOBS


class RuleScore:
    """No learning: average of name and address agreement signals (floor baseline)."""
    COLS = ["n_tset", "ns_lev", "sk_tset", "a_tset", "st_overlap", "num_overlap"]

    def fit(self, X, y, Xv=None, yv=None):
        return self

    def predict(self, X):
        return X[self.COLS].fillna(0.5).mean(1).values


class SkModel:
    """sklearn estimator with NaN handling (median impute + missing indicators)."""

    def __init__(self, est, scale=False):
        steps = [SimpleImputer(strategy="median", add_indicator=True)]
        if scale:
            steps.append(StandardScaler())
        self.pipe = make_pipeline(*steps, est)

    def fit(self, X, y, Xv=None, yv=None):
        self.pipe.fit(X.values, y)
        return self

    def predict(self, X):
        return self.pipe.predict_proba(X.values)[:, 1]


class LGBM:
    def __init__(self, **kw):
        import lightgbm as lgb
        self.lgb = lgb
        self.params = dict(n_estimators=3000, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           n_jobs=NJ, random_state=SEED, verbose=-1, **kw)

    def fit(self, X, y, Xv, yv):
        self.m = self.lgb.LGBMClassifier(**self.params)
        self.m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[self.lgb.early_stopping(100, verbose=False)])
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class XGB:
    def __init__(self, **kw):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=5,
                                   subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                   early_stopping_rounds=100, eval_metric="logloss",
                                   n_jobs=NJ, random_state=SEED, **kw)

    def fit(self, X, y, Xv, yv):
        self.m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


class CatB:
    def __init__(self, **kw):
        from catboost import CatBoostClassifier
        self.m = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=8, od_type="Iter", od_wait=100,
                                    thread_count=NJ, random_seed=SEED, verbose=False, **kw)

    def fit(self, X, y, Xv, yv):
        self.m.fit(X, y, eval_set=(Xv, yv))
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def zoo():
    """name -> factory. Each factory returns a fresh, unfitted model."""
    return {
        "rule": lambda: RuleScore(),
        "logreg": lambda: SkModel(LogisticRegression(C=1.0, max_iter=1000), scale=True),
        "extratrees": lambda: SkModel(ExtraTreesClassifier(n_estimators=200, min_samples_leaf=20,
                                                           max_features=0.5, n_jobs=NJ, random_state=SEED)),
        "randomforest": lambda: SkModel(RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                                               max_features=0.3, n_jobs=NJ, random_state=SEED)),
        "mlp": lambda: SkModel(MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=50,
                                             batch_size=2048, random_state=SEED), scale=True),
        "lightgbm": lambda: LGBM(),
        "xgboost": lambda: XGB(),
        "catboost": lambda: CatB(),
    }
