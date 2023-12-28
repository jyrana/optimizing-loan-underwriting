import argparse
import numpy as np

from models import XGBoost, LogisticRegression

from utils import load_dataset, preprocess
from config import FINAL_MODEL, FEATURE_SET


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', help='Path to the input file')
    parser.add_argument('--output_csv', help='Path to the output file')
    parser.add_argument('--model_path', help='Path to the trained model')

    args = parser.parse_args()

    dataset = load_dataset(args.input_csv)
    dataset = preprocess(dataset)

    model = None
    if FINAL_MODEL == "logistic":
        model = LogisticRegression()
    elif FINAL_MODEL == "xgb":
        model = XGBoost()

    model.load_trained_model(args.model_path)

    dataset["predicted_pd"] = model.predict(dataset[FEATURE_SET])
    dataset.loc[dataset[FEATURE_SET].isnull().any(axis=1), "predicted_pd"] = np.nan

    dataset["predicted_pd"].to_csv(args.output_csv)
