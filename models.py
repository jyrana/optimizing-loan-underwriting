import pickle
import xgboost as xgb
import statsmodels.formula.api as smf

from config import XGB_SAVE_PATH, LR_SAVE_PATH
import warnings

warnings.filterwarnings('ignore')
import pickle
class LogisticRegression:
    def __init__(self):
        self.model = None

    def fit_model(self, data, target, features):
        print('inside_fit_model')
        independent_vars_str = " + ".join(features[:-1])
        formula = f'{features[-1]} ~ {independent_vars_str}'
        print(formula)
        self.model = smf.logit(formula, data=data).fit(disp=False)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def save_model(self):
        with open(LR_SAVE_PATH, 'wb') as file:
            pickle.dump(self.model, file)

    def load_trained_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)


class XGBoost:

    def __init__(self,model_name):
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': 0.013
        }
        self.model_name = model_name
        self.model = None

    def fit_model(self, data, target, features):
        dtrain = xgb.DMatrix(data[features], label=data[target], enable_categorical=True)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)

    def predict(self, test_data):
        dtest = xgb.DMatrix(test_data, enable_categorical=True)
        return self.model.predict(dtest)

    def save_model(self):
        model_path = f"{self.model_name}_model.json"
        self.model.save_model(model_path)

    def load_trained_model(self, model_path):
        self.model = xgb.Booster(model_file=model_path)
        


