import os
from sklearn.linear_model import LogisticRegression
# from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

Config = {}

PACKAGE_DIR = os.path.dirname(os.path.abspath("__file__"))
_data_path = 'raw_data/avazu-ctr-prediction/'


Config['log_dir'] = os.path.join(PACKAGE_DIR, 'logs')


Config['data'] = {
    "train_path":  os.path.join(os.path.join(PACKAGE_DIR, _data_path), 'train.gz'),
    "test_path":  os.path.join(os.path.join(PACKAGE_DIR, _data_path), 'test.gz'),
}

Config['sample_data'] = os.path.join(PACKAGE_DIR, 'raw_data/sample.pkl')
Config['train_date'] = os.path.join(PACKAGE_DIR, 'raw_data/train.pkl')
Config['test_data'] = os.path.join(PACKAGE_DIR, 'raw_data/test.pkl')



Config['training_setting'] = {
    'num_threads': int(os.getenv("TRAINING_THREADS", 6)), 
    'num_workers': int(os.getenv("TRAINING_NUM_WORKERS", 10))
    }

           
FEATURES = ['site_id', 'site_domain', 'app_id', 'device_id', 'device_ip',\
    'device_model', 'C14', 'C1', 'banner_pos', 'device_type',\
    'device_conn_type', 'C15', 'C16', 'C18', 'site_category',\
    'C19', 'C21', 'app_category', 'C20', 'C17', 'app_domain',\
    'part_day', 'week']

ENCODER_LIST = ['woe', 'loo', 'catboost', 'target', 'count']
MODEL_LIST = ['Xgboost']
# MODEL_LIST = ['Xgboost', 'logisticReg', 'randomForest']


# parm_dict = {'Xgboost': [{
#                     'min_child_weight': [1, 5, 10],
#                     'gamma': [0.5, 1, 1.5, 2, 5],
#                     'subsample': [0.6, 0.8, 1.0],
#                     'colsample_bytree': [0.6, 0.8, 1.0],
#                     'max_depth': [3, 4, 5]
#                     }, XGBClassifier()],
#             'logisticReg': [{
#                     'penalty': ['l1', 'l2'],
#                     'C':[0.001, 0.01, 0.09, 1],
#                     'fit_intercept': [True, False]
#                     }, LogisticRegression()],
#             'randomForest': [{
#                     'n_estimators': [100, 300], 
#                     'max_features': ['auto', 'sqrt', 'log2'], 
#                     'max_depth': [1, 3, 5], 
#                     'criterion': ['gini', 'entropy']
#                     }, RandomForestClassifier()]
#             }


parms = [{
'n_estimators': [100, 300], 
'max_features': ['auto', 'sqrt', 'log2'], 
'max_depth' : [1, 3, 5], 
'criterion' :['gini', 'entropy']}, 
{"base_estimator__criterion" : ["gini", "entropy"],
"base_estimator__splitter" :   ["best", "random"],
"n_estimators": [1, 2, 3]}, 
{'penalty': ['l1', 'l2'],'C':[0.001, 0.01, 0.09, 1]},
{'C': [1, 3, 5], 'penalty': ['l2', 'l1']},
{'min_child_weight': [1, 5, 8],
'gamma': [0.5, 1, 1.5, 2],
'subsample': [0.6, 0.8, 1.0],''
'colsample_bytree': [0.6, 0.8, 1.0],
'max_depth': [3, 4, 5]},
{'learning_rate': [0.03, 0.1],
'depth': [4, 6, 10],
'l2_leaf_reg': [1, 3, 5, 7, 9]}
]