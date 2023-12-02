from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import dill
from src.config import sample_data
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None):
        return self

    def transform(self, X, y= None):
        X['datetime'] = X['hour'].astype(str).apply(lambda x: datetime.strptime(x, '%y%m%d%H'))
        X['dow'] =  X['datetime'].dt.dayofweek.apply(lambda x: x+1).astype(int)
        X['hod'] = X.datetime.apply(lambda x: x.hour)

        def hour_bin(data):
            data = int(data)
            if 5 < data <= 12:
                return 'Morning'
            elif 12 < data <= 17:
                return 'Afternoon'
            elif 17 < data < 21:
                return 'Evening'
            else:
                return 'Night'

        X['part_day'] = X.hod.apply(hour_bin)
        def hour_bin(data):
            data = int(data)
            if 1 <= data <= 5:
                return 'weekday'
            return 'weekend'

        X['week'] = X.dow.apply(hour_bin)
        dic = {c: X[c].nunique() for c in X.columns}
        unknown_feats = [c for c in X.filter(regex='C[\d]', axis=1).columns if dic[c] > 30]
        for col in unknown_feats:
            threshold = np.percentile(X[col].value_counts(), 75)
            tmp = [c for c in X[col].value_counts().index if X[col].value_counts()[c] <= threshold]
            X[col] = X[col].apply(lambda x:'others' if x in tmp else x)
        return X


if __name__ == "__main__":
    processor = Preprocessor()
    df = dill.load(open(sample_data, 'rb'))
    df = processor.transform(df)
    