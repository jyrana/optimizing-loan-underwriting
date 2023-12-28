TARGET = "is_default"

ROOT_PATH = ""

DATA_PATH = ROOT_PATH + "train.csv"

LR_SAVE_PATH = ROOT_PATH + "lr_model.pkl"
XGB_SAVE_PATH = ROOT_PATH + "xgb_model.json"

TRUE_PD = 0.0952
SAMPLE_PD = 0.01274

NORM_MEAN = 11163910.359835
NORM_STD = 182824801.73946536

DAYS_IN_YEAR = 365

FEATURE_SET = ["cash_ratio", "cash_roa", "debt_assets_lev", "avg_receivables_collection_day", "norm_asst_tot",
               "legal_struct", "ateco_sector", "HQ_city",'is_default']
FEATURE_SET_2 = ["current_ratio", "cash_roa", "debt_assets_lev", "asset_turnover", "norm_asst_tot",
               "legal_struct", "ateco_sector", "HQ_city",'is_default']
CATEGORICAL_FT = ["legal_struct", "ateco_sector", "HQ_city"]


FINAL_MODEL = "xgb"
