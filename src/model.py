import dill
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile
from src.preprocessing import Preprocessor
from src.encoding import create_encoder
from src.config import train_data, ENCODER_LIST, parm_dict, MODEL_LIST, models_dir
import pandas as pd

pd.set_option('mode.chained_assignment', None)

def train_flow(X_train, y_train, encoder, model):
    tmp_dict = parm_dict[model]
    param, estimator = tmp_dict[0], tmp_dict[1]

    pipeline = Pipeline([
        ('preprocessing', Preprocessor()),
        ('features', FeatureUnion(
            n_jobs= 4,
            transformer_list=[
                ("encoder", encoder),
            ])),
        ('feature_selection', SelectPercentile()),
        ('model', estimator)
    ])

    new_parm = {}
    for k, v in param.items():
            new_key = 'model__' + k
            new_parm[new_key] = v
    param = {
    'feature_selection__percentile': range(10, 110, 10),
    }
    new_parm.update(param)

    randomized_search_cv = RandomizedSearchCV(
        estimator= pipeline,
        param_distributions= new_parm,
        n_jobs= 4,
        verbose= 3,
        cv= TimeSeriesSplit(
            n_splits= 3,
            test_size= int(X_train.shape[0]/10)
        ),
        scoring= 'neg_log_loss',
        refit= True,
        n_iter= 3,
    )
    randomized_search_cv.fit(X_train, y_train)
    best_score = -randomized_search_cv.best_score_
    best_parameter = randomized_search_cv.best_estimator_.get_params()
    return randomized_search_cv, best_score, best_parameter

if __name__ == '__main__':
    train = dill.load(open(train_data, 'rb'))
    train = train.head(100000)
    X_train = train
    y_train = train["click"]

    for m in MODEL_LIST:
        for encoder_name in ENCODER_LIST:
            encoder = create_encoder(encoder_name)
            model, best_score, best_parameter = train_flow(X= X_train, y= y_train, \
                                                    encoder=encoder, model= m)
            dill.dump(
                obj=(model, best_score, best_parameter),
                file=open(models_dir + "/{}_{}.pkl".format(m, encoder_name), 'wb')
            )






