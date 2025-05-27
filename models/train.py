
import argparse
import pandas as pd
import mlflow, mlflow.xgboost
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='data/processed/features_model.parquet')
    p.add_argument('--test_size', type=float, default=0.2)
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.features)
    X = df.drop(['bookings','upsell_flag'], axis=1)
    y_reg = df['bookings']
    y_clf = df['upsell_flag']

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=args.test_size, shuffle=False
    )
    Xc_train, Xc_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=args.test_size, shuffle=False
    )

    mlflow.set_experiment('VacationOwnership')
    with mlflow.start_run(run_name='auto_run'):
        reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
        reg.fit(X_train, y_reg_train)
        mae = mean_absolute_error(y_reg_test, reg.predict(X_test))
        mlflow.log_param('reg_n_estimators', 100)
        mlflow.log_metric('mae', mae)
        mlflow.xgboost.log_model(reg, 'model_reg')

        clf = XGBClassifier(
            n_estimators=100, max_depth=6,
            use_label_encoder=False, eval_metric='logloss'
        )
        clf.fit(Xc_train, y_clf_train)
        auc = roc_auc_score(y_clf_test, clf.predict_proba(Xc_test)[:,1])
        mlflow.log_param('clf_n_estimators', 100)
        mlflow.log_metric('roc_auc', auc)
        mlflow.xgboost.log_model(clf, 'model_clf')

if __name__ == '__main__':
    main()
