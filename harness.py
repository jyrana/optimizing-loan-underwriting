import argparse
import numpy as np
import pandas as pd
from models import XGBoost, LogisticRegression

from utils import load_dataset, preprocess, calibrate
from config import FINAL_MODEL, FEATURE_SET, FEATURE_SET_2, TRUE_PD, SAMPLE_PD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', help='Path to the input file')
    parser.add_argument('--output_csv', help='Path to the output file')

    args = parser.parse_args()

    dataset = load_dataset(args.input_csv)
    
    dataset = preprocess(dataset)
    #print(dataset.current_ratio)
    model = None
    if FINAL_MODEL == "logistic":
        model = LogisticRegression()
        model.load_trained_model('lr_model.pkl')
        dataset['predicted_pd'] = model.predict[dataset[FEATURE_SET[:-1]].copy]
    elif FINAL_MODEL == "xgb":
        model = XGBoost('xgb')
        model.load_trained_model('xgb_1_model.json')
        dataset["predicted_pd_xgb1"] = model.predict(dataset[FEATURE_SET[:-1]].copy())
        model2 = XGBoost('xgb_2')
        model2.load_trained_model('xgb_2_model.json')
        dataset['predicted_pd_xgb2'] = model2.predict(dataset[FEATURE_SET_2[:-1]].copy())
   
        dataset['predicted_pd'] = (dataset['predicted_pd_xgb1'] + dataset.copy()['predicted_pd_xgb2'])/2
    

    dataset = calibrate(dataset, TRUE_PD, SAMPLE_PD)

    #as a last resort, instead of null, return PD of 0 since default is much less likely     
    dataset['predicted_pd'] = np.where(np.isnan(dataset['predicted_pd']), 0, dataset['predicted_pd'])
    
    dataset["predicted_pd_calibrated"].to_csv(args.output_csv, index=False, header = False)

