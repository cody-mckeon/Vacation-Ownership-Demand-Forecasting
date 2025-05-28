import pandas as pd
import mlflow, mlflow.xgboost
from xgboost import XGBClassifier

# Load & preprocess (as before)…
df = pd.read_parquet('data/processed/hotel_bookings_clean.parquet')
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue']      = df['adr'] * df['total_nights']
df['upsell_flag']  = (df['total_of_special_requests'] > 0).astype(int)

cat_cols = ['hotel','meal','market_segment','distribution_channel','deposit_type','customer_type']
num_cols = ['lead_time','total_nights','revenue']
# … build preprocessor & X as before …

# Split chronologically
from sklearn.model_selection import train_test_split
X_all, y_all = X, df['upsell_flag']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

best_params = {
    'n_estimators': 94,
    'max_depth': 10,
    'learning_rate': 0.05770112097140795,
    'subsample': 0.957552844824683,
    'colsample_bytree': 0.8113058633396639,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

mlflow.set_experiment("VacationOwnership_Upsell")
with mlflow.start_run(run_name="final_upsell_model"):
    # Log hyperparameters
    mlflow.log_params(best_params)
    
    # Train on the 80/20 split (or on all data if you prefer)
    clf = XGBClassifier(**best_params)
    clf.fit(X_train, y_train)
    
    # Evaluate on hold-out
    preds = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    mlflow.log_metric("holdout_auc", auc)
    
    # Log and register the model
    mlflow.xgboost.log_model(
        clf,
        artifact_path="upsell_model",
        registered_model_name="UpsellClassifier"
    )
    print(f"Registered UpsellClassifier with ROC AUC: {auc:.3f}")
