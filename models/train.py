import argparse
import pandas as pd
import mlflow, mlflow.xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--cleaned',
        default='data/processed/hotel_bookings_clean.parquet',
        help='Path to cleaned booking-level parquet'
    )
    p.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data to hold out for test'
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load cleaned booking-level data
    df = pd.read_parquet(args.cleaned)

    # 2) Recompute engineered columns
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['revenue']      = df['adr'] * df['total_nights']
    df['upsell_flag']  = (df['total_of_special_requests'] > 0).astype(int)

    # 3) Define features
    cat_cols = [
        'hotel','meal','market_segment',
        'distribution_channel','deposit_type','customer_type'
    ]
    num_cols = ['lead_time','total_nights','revenue']

    # 4) Build & apply preprocessor
    ohe = OneHotEncoder(
        drop='first', sparse_output=False, handle_unknown='ignore'
    )
    preprocessor = ColumnTransformer(
        transformers=[('cat', ohe, cat_cols)],
        remainder='passthrough'
    )
    X_all = preprocessor.fit_transform(df[cat_cols + num_cols])
    feature_names = (
        list(preprocessor.named_transformers_['cat']
             .get_feature_names_out(cat_cols))
      + num_cols
    )
    X = pd.DataFrame(X_all, columns=feature_names)

    # 5) Define targets
    y_reg = df['adr'] * df['total_nights']
    y_clf = df['upsell_flag']

    # 6) Chronological split
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=args.test_size, shuffle=False
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=args.test_size, shuffle=False
    )

    # --- Best params from Optuna tuning ---
    best_params = {
        'n_estimators': 94,
        'max_depth': 10,
        'learning_rate': 0.05770112097140795,
        'subsample': 0.957552844824683,
        'colsample_bytree': 0.8113058633396639,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }

    # 7) Train & log the tuned classifier
    mlflow.set_experiment("VacationOwnership_Upsell")
    with mlflow.start_run(run_name="final_upsell_model"):
        mlflow.log_params(best_params)
        clf = XGBClassifier(**best_params)
        clf.fit(X_train, y_clf_train)
        probs = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_clf_test, probs)
        mlflow.log_metric("holdout_auc", auc)
        mlflow.xgboost.log_model(
            clf,
            artifact_path="upsell_model",
            registered_model_name="UpsellClassifier"
        )
        print(f"Registered UpsellClassifier with ROC AUC: {auc:.3f}")

    # 8) Train & log the baseline regressor
    with mlflow.start_run(run_name="booking_level_regression"):
        reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        reg.fit(X_train, y_reg_train)
        preds = reg.predict(X_test)
        mae = mean_absolute_error(y_reg_test, preds)
        mlflow.log_metric("mae", mae)
        mlflow.xgboost.log_model(reg, "booking_reg")
        print(f"Booking-level MAE on revenue: {mae:.2f}")

if __name__ == "__main__":
    main()
