# Churn Prediction Streamlit App

A beginner-friendly churn prediction project using the BigML churn dataset.
Includes:
- EDA notebook (exploration + plots)
- Training script (reproducible model training + saved artifacts)
- Streamlit app (interactive predictions)

## Project Structure
- `notebooks/` - EDA notebook
- `src/train.py` - trains small + full models and saves artifacts to `models/`
- `app/app.py` - Streamlit UI that loads the small model
- `data/` - churn-bigml-80.csv (train) and churn-bigml-20.csv (test)
- `models/` - saved models + categories.json for dropbacks

## How to Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
