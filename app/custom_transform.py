from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AreaUnitConverter(BaseEstimator, TransformerMixin):
    def __init__(self, column='area'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        def convert_area(value):
            try:
                value = str(value).lower().replace(",", "").strip()
                if 'kanal' in value:
                    num = float(value.replace('kanal', '').strip())
                    return num * 20
                elif 'marla' in value:
                    return float(value.replace('marla', '').strip())
                else:
                    return float(value)
            except:
                return np.nan
        
        X[self.column] = X[self.column].apply(convert_area)
        return X
