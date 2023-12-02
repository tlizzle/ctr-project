import dill
from sklearn.metrics import log_loss
from src.config import MODEL_LIST, ENCODER_LIST,\
                    test_data, models_dir

def evaluate_flow(X_test, y_test, model):
    y_predict_proba = model.predict_proba(X_test)[:, 1]
    log_loss_ = log_loss(y_true= y_test, y_pred= y_predict_proba)
    print(m, encoder_name, best_score_, log_loss_)
        
if __name__ == '__main__':
    test = dill.load(open(test_data, 'rb'))
    X_test = test
    y_test = test["click"]

    for m in MODEL_LIST:
        for encoder_name in ENCODER_LIST:
            model, best_score_, best_parameters = \
                    dill.load(open(models_dir + "/{}_{}.pkl".format(m, encoder_name), 'rb'))
            evaluate_flow(X_test, y_test, model)
