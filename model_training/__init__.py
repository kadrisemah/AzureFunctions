import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
import azure.functions as func

warnings.filterwarnings('ignore')


def create_db_connection(server, database, username, password):
    driver = 'ODBC Driver 18 for SQL Server'
    connection_string = (
        f'mssql+pyodbc://{username}:{quote_plus(password)}@{server}:1433/{database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    return create_engine(connection_string)


def main(mytimer: func.TimerRequest) -> None:
    logging.info('Model Training Function started')

    INPUT_SCHEMA = 'dbo'
    INPUT_TABLE = 'topup_training_dataset'
    TRAIN_CUTOFF = '2024-06-01'
    TEST_MONTH = '2025-07-01'
    PROBABILITY_THRESHOLD = 0.5

    logging.info('Retrieving credentials from application settings...')

    db_server = os.environ.get('TARGET_SERVER')
    db_database = os.environ.get('TARGET_DATABASE')
    db_username = os.environ.get('TARGET_USER')
    db_password = os.environ.get('TARGET_PASSWORD')

    if not all([db_server, db_database, db_username, db_password]):
        logging.error('Missing required environment variables. Please configure TARGET_* credentials in Application Settings.')
        raise ValueError('Missing required environment variables')

    logging.info('Credentials retrieved successfully')

    logging.info('Establishing database connection...')
    engine = create_db_connection(db_server, db_database, db_username, db_password)
    logging.info('Database connection established')

    logging.info(f'Loading data from {INPUT_SCHEMA}.{INPUT_TABLE}...')
    query = f'SELECT * FROM {INPUT_SCHEMA}.{INPUT_TABLE}'
    data = pd.read_sql(query, engine)
    logging.info(f'Loaded {len(data)} rows, {len(data.columns)} columns')
    logging.info(f'Date range: {data["month"].min()} to {data["month"].max()}')
    logging.info(f'Unique clients: {data["HubSpotId"].nunique()}')

    logging.info('Creating target variable...')
    if 'target_next_month_TOPUP' in data.columns:
        data['target_flag'] = (data['target_next_month_TOPUP'].fillna(0) > 0).astype(int)
    else:
        data = data.sort_values(['HubSpotId', 'month'])
        data['target_flag'] = data.groupby('HubSpotId')['TOPUP'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0 else 0)

    target_dist = data['target_flag'].value_counts()
    logging.info(f'Target distribution: {dict(target_dist)}')
    logging.info(f'Positive rate: {data["target_flag"].mean():.2%}')

    logging.info('Preparing features...')
    exclude_cols = ['HubSpotId', 'month', 'TOPUP', 'TOPUPAMOUNT', 'target_next_month_TOPUP', 'target_flag', 'last_contact_date', 'month_end', 'created_at', 'pipeline_run_id']
    numeric_cols = data.select_dtypes(include=['number']).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    categorical_cols = ['RiskProfile', 'PoA', 'season', 'EngagementLabel', 'AgeRange', 'Gender', 'ShariaCompliant', 'month_name']
    categorical_cols = [c for c in categorical_cols if c in data.columns]

    for col in categorical_cols:
        if col not in feature_cols:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown').astype(str))
            feature_cols.append(f'{col}_encoded')

    feature_cols_with_meta = feature_cols + ['month', 'HubSpotId', 'target_flag']
    logging.info(f'Selected {len(feature_cols)} features')

    logging.info('Detecting outliers...')
    contamination = 0.05
    X_outlier = data[feature_cols].fillna(0)
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_labels = iso_forest.fit_predict(X_outlier)
    data['is_outlier'] = (outlier_labels == -1).astype(int)
    n_outliers = data['is_outlier'].sum()
    logging.info(f'Detected {n_outliers} outliers ({n_outliers/len(data):.2%})')

    logging.info(f'Splitting data: train <= {TRAIN_CUTOFF}, test = {TEST_MONTH}...')
    data['month_str'] = pd.to_datetime(data['month']).dt.strftime('%Y-%m-%d')
    train_df = data[data['month_str'] <= TRAIN_CUTOFF].copy()
    test_df = data[data['month_str'] == TEST_MONTH].copy()

    if 'is_outlier' in train_df.columns:
        original_train_size = len(train_df)
        train_df = train_df[train_df['is_outlier'] == 0]
        logging.info(f'Removed {original_train_size - len(train_df)} outliers from training')

    leak_cols = ['target_next_month_TOPUP', 'target_flag', 'month', 'month_str', 'HubSpotId', 'is_outlier', 'TOPUP', 'TOPUPAMOUNT', 'created_at', 'pipeline_run_id']
    feature_cols_clean = [c for c in feature_cols if c not in leak_cols]

    X_train = train_df[feature_cols_clean].fillna(0)
    y_train = train_df['target_flag']
    X_test = test_df[feature_cols_clean].fillna(0)
    y_test = test_df['target_flag']
    test_hubspot_ids = test_df['HubSpotId'].values

    logging.info(f'Train: {len(X_train)} samples, {len(X_train.columns)} features')
    logging.info(f'Train positive rate: {y_train.mean():.2%}')
    logging.info(f'Test: {len(X_test)} samples')
    logging.info(f'Test positive rate: {y_test.mean():.2%}')

    cat_features = [i for i, c in enumerate(X_train.columns) if c.endswith('_encoded')]
    logging.info(f'Categorical features: {len(cat_features)}')

    logging.info('Training CatBoost model...')
    n0, n1 = np.bincount(y_train)
    class_weights = {0: 1.0, 1: n0 / max(n1, 1)}

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.001,
        depth=16,
        l2_leaf_reg=5,
        bagging_temperature=0.2,
        subsample=0.8,
        rsm=0.8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        class_weights={0:1.0, 1: 0.95},
        od_type='Iter',
        early_stopping_rounds=100,
        verbose=100
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)
    logging.info('Model training completed')

    logging.info('Evaluating model...')
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    test_predictions = pd.DataFrame({
        'HubSpotId': test_hubspot_ids,
        'y_true': y_test.values,
        'y_pred': y_pred_test,
        'y_proba': y_proba_test
    })

    logging.info('TRAIN SET PERFORMANCE')
    logging.info(classification_report(y_train, y_pred_train))
    logging.info(f'ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}')

    logging.info('TEST SET PERFORMANCE')
    logging.info(classification_report(y_test, y_pred_test))
    logging.info(f'ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}')

    logging.info('Analyzing feature importance...')
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logging.info('Top 20 Feature Importances:')
    logging.info(importance_df.head(20).to_string(index=False))

    model_filename = f'/tmp/catboost_topup_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.cbm'
    model.save_model(model_filename)
    logging.info(f'Model saved: {model_filename}')

    logging.info(f'Generating action list (threshold={PROBABILITY_THRESHOLD})...')
    action_list = test_predictions[test_predictions['y_proba'] >= PROBABILITY_THRESHOLD].sort_values('y_proba', ascending=False).copy()
    logging.info(f'Action list: {len(action_list)} high-probability clients')
    logging.info(f'Average probability: {action_list["y_proba"].mean():.2%}')
    logging.info('Top 10 Clients:')
    logging.info(str(action_list.head(10)))

    logging.info('Saving results to SQL...')
    test_predictions.to_sql('topup_model_predictions', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Predictions saved to {INPUT_SCHEMA}.topup_model_predictions')

    action_list.to_sql('topup_rm_action_list', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Action list saved to {INPUT_SCHEMA}.topup_rm_action_list')

    importance_df.to_sql('topup_feature_importance', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Feature importance saved to {INPUT_SCHEMA}.topup_feature_importance')

    logging.info('=' * 70)
    logging.info('PIPELINE COMPLETED SUCCESSFULLY')
    logging.info('=' * 70)
    logging.info(f'Training samples: {len(X_train)}')
    logging.info(f'Test samples: {len(X_test)}')
    logging.info(f'Features: {len(X_train.columns)}')
    logging.info(f'Train ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}')
    logging.info(f'Test ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}')
    logging.info(f'High-probability clients: {len(action_list)}')
    logging.info(f'Model saved: {model_filename}')
    logging.info('=' * 70)
