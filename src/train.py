from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# BASE_DIR for the repository root (../ from src/train.py).
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "churn-bigml-80.csv"
TEST_PATH = BASE_DIR / "data" / "churn-bigml-20.csv"

# Where we save trained models for the Streamlit APP.
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# The Streamlit APP collects only these 4 inputs, so the "small" model is trained
cate_cols = ["State", "Area code", "International plan", "Voice mail plan"]

# Load train/test data (80/20 split files from the BigML churn dataset)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["Churn"]
y_test = test_df["Churn"]

#======================================================================
# Small Model (4 inputs only) - used by the Streamlit APP
#======================================================================
X_train_small = train_df[cate_cols].copy()
X_test_small = test_df[cate_cols].copy()


# - OneHotEncode the categorical fields
# - handle_unknown="ignore" prevents errors if a new category appears
# - drop="if_binary" prevents redanunce with yes/no columns
# - remainder="drop" ensures the model uses ONLY the 4 categerieos
small_prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cate_cols),remainder="drop")

small_logreg = LogisticRegression(max_iter=1000)
small_pipe = make_pipeline(small_prep, small_logreg)

small_pipe.fit(X_train_small, y_train)
small_pred = small_pipe.predict(X_test_small)

small_acc = accuracy_score(y_test, small_pred)
small_f1 = f1_score(y_test, small_pred)

print("SMALL MODEL")
print("Accuracy: ", small_acc)
print("F1: ", small_f1)

# Save trained pipeline for Streamlit APP
joblib.dump(small_pipe, MODEL_DIR / "churn_pipe_small.joblib")

# Save categories for Streamlit APP (dropback)
ct = small_pipe.named_steps["columntransformer"]
ohe = ct.named_transformers_["onehotencoder"]
cats = {col: [str(v) for v in values] for col, values in zip(cate_cols, ohe.categories_)}

with open(MODEL_DIR / "categories.json", "w", encoding="utf-8") as f:
    json.dump(cats, f, indent=2)

#==============================================================================================
# Full Model - better performance
#==============================================================================================
X_train_full = train_df.drop(columns=["Churn"])
X_test_full = test_df.drop(columns=["Churn"])

# - OneHotEncode the categorical fields
# - handle_unknown="ignore" prevents errors if a new category appears
# - drop="if_binary" prevents redanunce with yes/no columns
# - remainder="passtrough" considers all categories into the model
full_prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore",drop="if_binary"), cate_cols), remainder="passthrough")

full_logreg = LogisticRegression(max_iter=1000)
full_pipe = make_pipeline(full_prep, full_logreg)

full_pipe.fit(X_train_full, y_train)
full_pred = full_pipe.predict(X_test_full)

full_acc = accuracy_score(y_test, full_pred)
full_f1 = f1_score(y_test, full_pred)

print("FULL MODEL")
print("Accuracy:", full_acc)
print("F1:", full_f1)

# Save trained pipeline for future use
joblib.dump(full_pipe, MODEL_DIR / "churn_pipe.joblib")