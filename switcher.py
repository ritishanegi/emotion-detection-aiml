#!/usr/bin/env python3

'''
ClfSwitcher for scikit-learn Pipeline and GridSearchCV support.
'''

from sklearn.base import BaseEstimator, ClassifierMixin

class ClfSwitcher(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        if self.estimator is None:
            return self
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return self.estimator.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.estimator.predict_proba(X, **kwargs)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=sample_weight)
