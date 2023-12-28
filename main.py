import argparse

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

from config import *
from features import FeatureSelection
from models import *
from utils import load_dataset, assert_low_VIF
from preprocess import Preprocessor
import warnings


def plot(pred_list, est_name):
    # Combine predictions and true labels from all walk-forward steps
    pred_df = pd.concat(pred_list)

    all_predictions = pred_df['y_pred_prob']
    all_true_labels = pred_df['y_true']

    # Calculate the ROC curve
    fpr, tpr, _ = metrics.roc_curve(all_true_labels, all_predictions)

    # Calculate the AUC
    roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{est_name} ROC")
    plt.legend(loc='lower right')
    plt.show()

    # Display the AUC value
    print(f'\nAUC: {roc_auc:.6f}')


class WalkForward:

    def __init__(self, dataset):

        self.dataset = dataset
        self.target = TARGET

        self.model = None
        self.train_mean, self.train_std = None, None

    def __normalize_assets(self, train_df, test_df):
        """
        The total asset values in both train and test datasets are normalized
        using mean and std from ONLY the train dataset
        Args:
            train_df: the train slice
            test_df: the test slice
        Returns:
            train and test slices with asst_tot normalized
        """
        self.train_mean = train_df["asst_tot"].mean()
        self.train_std = train_df["asst_tot"].std()

        train_df["norm_asst_tot"] = (train_df["asst_tot"] - self.train_mean) / self.train_std
        test_df["norm_asst_tot"] = (test_df["asst_tot"] - self.train_mean) / self.train_std
        return train_df, test_df

    def __predictor(self, est_name,test_df, features,scaler):
        pred_df = pd.DataFrame()

        if not test_df.empty:
            
            pred_df['y_true'] = test_df[self.target].copy()
            pred_df['y_pred_prob'] = self.model.predict(test_df[features])

        return pred_df

    def __estimator(self, est_name, features, data, model_name):
        if est_name == "logistic":
            self.model = LogisticRegression()
            self.model.fit_model(data, self.target, features)
            return None
        elif est_name == "xgb":
            self.model = XGBoost(model_name)
            self.model.fit_model(data, self.target, features)
            return None
    def __walk_forward(self, est_name, features,model_name, start_year):
        """
        Performs training using the walk-forward approach.
        Args:
            est_name: the estimator to use
            features: the features to use for training
            start_year: the year to start the walk-forward analysis from
        Returns:
            list of all walk-forward models and their corresponding predictions
        """
        max_year = self.dataset['fs_year'].max()

        model_list, pred_list = [], []

        for year in range(start_year, max_year+1):
            self.model = None
            train_df = self.dataset[self.dataset['fs_year'] <= year].copy()
            test_df = self.dataset[self.dataset['fs_year'] == year+1].copy()
            train_df, test_df = self.__normalize_assets(train_df, test_df)

            default_rate = train_df[train_df[self.target] == 1].shape[0] / train_df.shape[0]
            print(f"\nYear: {year} | Sample default rate: {default_rate * 100:.3f}%")

            scaler = self.__estimator(est_name, features, train_df,model_name)
            if len(test_df)<1:
              continue
            pred = self.__predictor(est_name,test_df, features, scaler)

            model_list.append(self.model)
            pred_list.append(pred)

            del train_df, test_df

        return model_list, pred_list

    def run(self, est_name, features,model_name, start_year=2008):
        model_list, pred_list = self.__walk_forward(est_name, features, model_name,start_year)

        self.model.save_model()
        plot(pred_list, est_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', help='Path to the input file')
    parser.add_argument('--est_name', help='one of logistic, NN, xgb')
    parser.add_argument('--extract_features', action='store_true', help='To enable/disable feature extraction')

    args = parser.parse_args()

    data_path = args.input_csv if args.input_csv is not None else DATA_PATH
    dataset = load_dataset(data_path)

    preprocessor = Preprocessor()
    fs = FeatureSelection()

    dataset = preprocessor.preprocess(dataset)

    if args.extract_features:
        uni_variate_ft = fs.univariate_analysis(dataset)
        assert_low_VIF(dataset[uni_variate_ft])

        rfe_features = fs.rfe_analysis(dataset)
        assert_low_VIF(dataset[rfe_features])

    estimator_name = args.est_name if args.est_name else FINAL_MODEL
    wf = WalkForward(dataset)
    if estimator_name == 'logistic':
        wf.run(estimator_name, FEATURE_SET,'logit')
    else:
        wf.run(estimator_name, FEATURE_SET_2[:-1],'xgb_2')
        wf.run(estimator_name, FEATURE_SET[:-1],'xgb_1')
    