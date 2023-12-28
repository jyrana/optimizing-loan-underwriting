# Predicting the Probability of Default

## Files in the project

* **preprocess.py**: dataset preprocessing functions
* **features.py**: feature extraction based on Uni-variate Analysis and Recursive Feature Elimination (RFE)
* **models.py**: contains implementation LogisticRegression, NeuralNetwork and XGBoost Classifier
* **utils.py**: util functions
* **configs.py**: project configs
* **main.py**: driver functions for walk-forward analysis
* **harness.py**: inference

- - - -

## Usage

### For Training
`harness.py`

```commandline
python3 main.py --input_csv  <input file in csv> --extract_features <run feature extraction (bool)> --est_name <model to use>
```

### For Inference
`harness.py`

`Note:` Only the XGBoost model is up and running for the purpose of this test

```commandline
python3 harness.py --input_csv  <input file in csv> --output_csv <output csv file path to which the predictions are written>
```

