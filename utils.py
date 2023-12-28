import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from config import FEATURE_SET, FEATURE_SET_2, CATEGORICAL_FT, NORM_STD, NORM_MEAN


def load_dataset(data_path):
    dataset = pd.read_csv(data_path)
    if "Unnamed: 0" in dataset.columns:
        dataset = dataset.drop("Unnamed: 0", axis=1).reset_index(drop=True)
    return dataset


# -------------------------------------- Preprocessor Util Functions --------------------------------------------------#

def get_sector(sector):

    if sector in range(1, 4):
        return 1
    elif sector in range(5, 10):
        return 2
    elif sector in range(10, 13):
        return 3
    elif sector in range(13, 16):
        return 4
    elif sector in range(16, 19):
        return 5
    elif sector == 19:
        return 6
    elif sector == 20:
        return 7
    elif sector == 21:
        return 8
    elif sector in range(22, 24):
        return 9
    elif sector in range(24, 26):
        return 10
    elif sector == 26:
        return 11
    elif sector == 27:
        return 12
    elif sector == 28:
        return 13
    elif sector in range(29, 31):
        return 14
    elif sector in range(31, 34):
        return 15
    elif sector == 35:
        return 16
    elif sector in range(36, 40):
        return 17
    elif sector in range(41, 44):
        return 18
    elif sector in range(45, 48):
        return 19
    elif sector in range(49, 54):
        return 20
    elif sector in range(55, 57):
        return 21
    elif sector in range(58, 61):
        return 22
    elif sector == 61:
        return 23
    elif sector in range(62, 64):
        return 24
    elif sector in range(64, 67):
        return 25
    elif sector == 68:
        return 26
    elif sector in range(69, 72):
        return 27
    elif sector == 72:
        return 28
    elif sector in range(73, 76):
        return 29
    elif sector in range(77, 83):
        return 30
    elif sector == 84:
        return 31
    elif sector == 85:
        return 32
    elif sector == 86:
        return 33
    elif sector in range(87, 89):
        return 34
    elif sector in range(90, 94):
        return 35
    elif sector in range(94, 97):
        return 36
    elif sector in range(97, 99):
        return 37
    elif sector == 99:
        return 38
    else:
        return 'Missing'


def to_categorical(df):
    df['ateco_sector'] = df['ateco_sector'].apply(get_sector)
    df[CATEGORICAL_FT] = df[CATEGORICAL_FT].astype("category")
    return df


def preprocess(df):
    """
    Extracting required features from the dataset
    """

    df = to_categorical(df)

    # Existing features
    df["rev_operating"].fillna(df["prof_operations"] + df["COGS"], inplace=True)

    # New features
    df["cash_roa"] = df["cf_operations"] / df["asst_tot"]
    df["debt_assets_lev"] = (df["asst_tot"] - df["eqty_tot"]) / df["asst_tot"]
    df["cash_ratio"] = df["cash_and_equiv"] / df["debt_st"]
    df["asset_turnover"] = df["rev_operating"] / df["asst_tot"]
    df["norm_asst_tot"] = (df["asst_tot"] - NORM_MEAN) / NORM_STD
    df["receivable_turnover"] = df["rev_operating"] / df["AR"]
    df["avg_receivables_collection_day"] = 365 / df["receivable_turnover"]
    df["current_ratio"] = df["asst_current"] / df["debt_st"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

def calibrate(df, true_pd, sample_pd):
    import statsmodels.api as sm
    """
    Step 1: non-parametric mapping
    Step 2: Elkan method of calibration
    """

    #smoothing function
    lowess = sm.nonparametric.lowess

    calibration_df = df.copy()
    calibration_df.sort_values(ascending=True, by = 'predicted_pd', inplace=True)
    calibration_df.reset_index(inplace=True,drop = False)
    calibration_df.rename(columns={'index': 'sorter'}, inplace=True)

    z = lowess(calibration_df.predicted_pd, calibration_df.index, frac= 0.01, delta=len(calibration_df)*0.01)

    calibration_df['new_prediction'] = z[:, 1]
    calibration_df.sort_values(ascending=True, by = 'sorter', inplace=True)
    calibration_df.set_index('sorter', inplace=True, drop=True)

    #p is pd from model output
    p = calibration_df.new_prediction

    #elkan formula
    p_calibrated = true_pd * ((p - p*sample_pd)/(sample_pd - p*sample_pd + p*true_pd - sample_pd*true_pd))

    #return df with new calibrated col
    df['predicted_pd_calibrated'] = p_calibrated

    return df

# ----------------------------------- Feature Extraction Util Functions -----------------------------------------------#

def assert_low_VIF(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    assert sum(vif_data["VIF"] > 2) == 0


