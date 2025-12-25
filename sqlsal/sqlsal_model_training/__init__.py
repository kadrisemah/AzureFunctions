import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.samplers import TPESampler
import warnings
import os
import re
import azure.functions as func
import shap

warnings.filterwarnings('ignore')


def create_db_connection(server, database, username, password):
    """Create database connection using ODBC Driver 18 or 17"""
    driver = 'ODBC Driver 18 for SQL Server'
    connection_string = (
        f'mssql+pyodbc://{username}:{quote_plus(password)}@{server}:1433/{database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    return create_engine(connection_string)


def sanitize_column_names(df, verbose=True):
    """Remove invalid characters from column names for CatBoost compatibility"""
    new_columns = {}
    for col in df.columns:
        # Replace spaces and special characters with underscores
        new_col = re.sub(r'[^\w]', '_', col)
        # Remove consecutive underscores
        new_col = re.sub(r'_+', '_', new_col)
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        if new_col != col and verbose:
            logging.debug(f'Renamed column: {col} -> {new_col}')
        new_columns[col] = new_col
    df.rename(columns=new_columns, inplace=True)
    return df


def prepare_features(data):
    """Prepare features for model training"""
    logging.info('Preparing features for modeling...')

    # Columns to drop (non-features, target, metadata)
    columns_to_drop = [
        "GAA23PriorityPassRequested", 'LeadScoreProspectJourney', "GAA23PriorityPass",
        'EstimatedClientSegment', 'Team', 'Flow', 'HashID', 'CreateDay', "BirthYear",
        "ExpectedClosingProbability", 'SourceExtractionDate', 'HsPersona', 'PK_HS_Contact',
        'ProspectJourneyQualificationStatus', 'SourceOfFundsConsolidated',
        'CloseDate', 'HS_DaysToClose', 'DealCloseDateCommittedCapitalContact',
        'WeekNumber', 'YearNumber', 'LeadStageNewCEO', 'LeadStageNew', 'TotalDealAmount',
        'ExistingMappedLeadStages', 'LeadStage', 'OnboardingId', 'TotalDealAmountDS',
        'ExpectedMonthOfclosing', 'GAA23ListMembership', 'GAA23InviteeType', 'GAA23EntryDate',
        'HSLeadStatus', 'IPCountry', 'BirthDate', 'InterestedFlow',
        'PropensityScoreDFOSALs', 'GAA23EntryTime', 'GAA23CheckedInBy',
        'HubSpotId', 'CreateDate', 'ContactType', 'created_at', 'pipeline_run_id',
        'Target', 'ContactOwnerId', 'DAYSTOCLOSE', 'ValidFrom', 'ValidUntil',
        'MeetingNotesPreview', 'CallNotesPreview', 'SegmentClass',
        'Country', 'FirstPageSeen', 'Nationality', 'EngagementLabel', 'AgeRange',
        'CreateDate_Bins', 'City'
    ]

    # Keep metadata columns for later use
    metadata_cols = ['HubSpotId', 'CreateDate', 'ContactType', 'ContactOwnerId', 'LeadStageNewCEO']
    metadata_df = data[[col for col in metadata_cols if col in data.columns]].copy()

    # Extract target
    if 'Target' not in data.columns:
        logging.error('Target column not found in dataset')
        raise ValueError('Target column missing')

    y = data['Target'].copy()

    # Drop non-feature columns
    existing_drop_cols = [col for col in columns_to_drop if col in data.columns]
    X = data.drop(columns=existing_drop_cols, errors='ignore')

    # Sanitize column names
    X = sanitize_column_names(X, verbose=False)

    # Remove duplicate columns after sanitization (keeps first occurrence)
    original_cols = len(X.columns)
    X = X.loc[:, ~X.columns.duplicated()]
    if len(X.columns) < original_cols:
        logging.info(f'Removed {original_cols - len(X.columns)} duplicate columns after sanitization')

    # Fill missing values
    X = X.fillna(0)

    # Select only numeric features (one-hot encoded and numeric)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    logging.info(f'Features prepared: {X.shape[1]} features, {X.shape[0]} samples')
    logging.info(f'Target distribution: {y.value_counts().to_dict()}')

    return X, y, metadata_df


def create_optuna_objective(X_train, y_train, categorical_features):
    """Create Optuna objective function for hyperparameter optimization"""

    def objective(trial):
        # Suggest hyperparameters
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False
        }

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=categorical_features)
            val_pool = Pool(X_val, y_val, cat_features=categorical_features)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)

        return np.mean(scores)

    return objective


def main(mytimer: func.TimerRequest) -> None:
    logging.info('SQLSAL Model Training Function started')

    INPUT_SCHEMA = 'dbo'
    INPUT_TABLE = 'sqlsal_training_dataset'
    PROBABILITY_THRESHOLD = 0.5
    N_OPTUNA_TRIALS = 50  # Reduced from notebook for production
    TEST_SIZE = 0.2

    logging.info('Retrieving credentials from application settings...')

    db_server = os.environ.get('TARGET_SERVER')
    db_database = os.environ.get('TARGET_DATABASE')
    db_username = os.environ.get('TARGET_USER')
    db_password = os.environ.get('TARGET_PASSWORD')

    if not all([db_server, db_database, db_username, db_password]):
        logging.error('Missing required environment variables')
        raise ValueError('Missing required environment variables')

    logging.info('Credentials retrieved successfully')

    logging.info('Establishing database connection...')
    engine = create_db_connection(db_server, db_database, db_username, db_password)
    logging.info('Database connection established')

    logging.info(f'Loading data from {INPUT_SCHEMA}.{INPUT_TABLE}...')
    query = f'SELECT * FROM {INPUT_SCHEMA}.{INPUT_TABLE}'
    data = pd.read_sql(query, engine)
    logging.info(f'Loaded {len(data)} rows, {len(data.columns)} columns')
    logging.info(f'Unique contacts: {data["HubSpotId"].nunique()}')

    # Prepare features
    X, y, metadata_df = prepare_features(data)

    # Train-test split
    logging.info(f'Splitting data: {TEST_SIZE*100}% test set...')
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata_df, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    logging.info(f'Train: {len(X_train)} samples, {len(X_train.columns)} features')
    logging.info(f'Train positive rate: {y_train.mean():.2%}')
    logging.info(f'Test: {len(X_test)} samples')
    logging.info(f'Test positive rate: {y_test.mean():.2%}')

    # No categorical features in this case (all one-hot encoded)
    categorical_features = []

    # Hyperparameter optimization with Optuna
    logging.info(f'Starting Optuna hyperparameter optimization ({N_OPTUNA_TRIALS} trials)...')
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    objective = create_optuna_objective(X_train, y_train, categorical_features)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    logging.info(f'Best ROC-AUC from Optuna: {best_score:.4f}')
    logging.info(f'Best parameters: {best_params}')

    # Train final model with best parameters
    logging.info('Training final CatBoost model with best parameters...')
    final_params = {
        **best_params,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 100
    }

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    model = CatBoostClassifier(**final_params)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100, use_best_model=True)
    logging.info('Model training completed')

    # Evaluate model
    logging.info('Evaluating model...')
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    logging.info('TRAIN SET PERFORMANCE')
    logging.info(f'Accuracy: {accuracy_score(y_train, y_pred_train):.4f}')
    logging.info(f'Precision: {precision_score(y_train, y_pred_train):.4f}')
    logging.info(f'Recall: {recall_score(y_train, y_pred_train):.4f}')
    logging.info(f'F1-Score: {f1_score(y_train, y_pred_train):.4f}')
    logging.info(f'ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}')

    logging.info('TEST SET PERFORMANCE')
    logging.info(f'Accuracy: {accuracy_score(y_test, y_pred_test):.4f}')
    logging.info(f'Precision: {precision_score(y_test, y_pred_test):.4f}')
    logging.info(f'Recall: {recall_score(y_test, y_pred_test):.4f}')
    logging.info(f'F1-Score: {f1_score(y_test, y_pred_test):.4f}')
    logging.info(f'ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}')

    # Generate out-of-fold predictions for ALL data (prevents data leakage)
    logging.info('Generating out-of-fold predictions for all contacts...')
    oof_predictions = np.zeros(len(X))
    oof_predictions_proba = np.zeros(len(X))
    models = []  # Store all fold models

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f'Training Fold {fold}/5 for OOF predictions...')

        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model on this fold with best parameters
        fold_params = {
            **best_params,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False
        }
        fold_model = CatBoostClassifier(**fold_params)
        fold_pool_train = Pool(X_fold_train, y_fold_train, cat_features=categorical_features)
        fold_pool_val = Pool(X_fold_val, y_fold_val, cat_features=categorical_features)

        fold_model.fit(fold_pool_train, eval_set=fold_pool_val, early_stopping_rounds=100, use_best_model=True, verbose=False)

        # Predict on validation fold (unseen data for this model)
        oof_predictions[val_idx] = fold_model.predict(X_fold_val)
        oof_predictions_proba[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

        # Store model
        models.append(fold_model)

        # Fold metrics
        fold_auc = roc_auc_score(y_fold_val, oof_predictions_proba[val_idx])
        logging.info(f'Fold {fold} AUC: {fold_auc:.4f}')

    # Overall OOF metrics
    overall_oof_auc = roc_auc_score(y, oof_predictions_proba)
    logging.info(f'Overall OOF ROC-AUC: {overall_oof_auc:.4f}')

    # Use OOF predictions for all downstream tasks
    y_proba_all = oof_predictions_proba
    y_pred_all = oof_predictions

    # Create predictions DataFrame
    predictions_df = metadata_df.copy()
    predictions_df['y_true'] = y.values
    predictions_df['y_pred'] = y_pred_all
    predictions_df['probability'] = y_proba_all
    predictions_df['prediction_date'] = datetime.now()

    # Feature importance
    logging.info('Analyzing feature importance...')
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logging.info('Top 20 Feature Importances:')
    logging.info(importance_df.head(20).to_string(index=False))

    # SHAP Explainability
    logging.info('Generating SHAP values for model explainability...')
    try:
        # Use TreeExplainer for CatBoost (use the first fold model)
        explainer = shap.TreeExplainer(models[0])

        # Calculate SHAP values on a sample (for performance)
        sample_size = min(100, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)

        # Save SHAP summary statistics
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        logging.info(f'SHAP analysis completed ({len(shap_importance)} features)')
        logging.info('Top 10 SHAP Feature Importances:')
        logging.info(shap_importance.head(10).to_string(index=False))

    except Exception as e:
        logging.warning(f'SHAP analysis failed: {e}')
        # Create empty dataframe if SHAP fails
        shap_importance = pd.DataFrame(columns=['feature', 'mean_abs_shap'])

    # Save model
    model_filename = f'/tmp/catboost_sqlsal_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.cbm'
    model.save_model(model_filename)
    logging.info(f'Model saved: {model_filename}')

    # Generate action list
    logging.info(f'Generating action list (threshold={PROBABILITY_THRESHOLD})...')
    action_list = predictions_df[predictions_df['probability'] >= PROBABILITY_THRESHOLD].sort_values('probability', ascending=False).copy()

    # Filter for SQL contacts only (as per notebook)
    if 'LeadStageNewCEO' in action_list.columns:
        action_list = action_list[action_list['LeadStageNewCEO'].str.contains('SQL', case=False, na=False)]

    logging.info(f'Action list: {len(action_list)} high-probability SQL contacts')
    if len(action_list) > 0:
        logging.info(f'Average probability: {action_list["probability"].mean():.2%}')
        logging.info(f'Probability range: {action_list["probability"].min():.2%} - {action_list["probability"].max():.2%}')

    # Save results to SQL
    logging.info('Saving results to SQL...')

    # 1. All predictions
    predictions_output = predictions_df[['HubSpotId', 'y_true', 'y_pred', 'probability', 'prediction_date']].copy()
    predictions_output.to_sql('sqlsal_model_predictions', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Predictions saved to {INPUT_SCHEMA}.sqlsal_model_predictions ({len(predictions_output)} rows)')

    # 2. Action list for sales team
    action_list_output = action_list[['HubSpotId', 'CreateDate', 'ContactType', 'probability', 'ContactOwnerId']].copy() if 'ContactOwnerId' in action_list.columns else action_list[['HubSpotId', 'CreateDate', 'ContactType', 'probability']].copy()
    action_list_output.to_sql('sqlsal_rm_action_list', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Action list saved to {INPUT_SCHEMA}.sqlsal_rm_action_list ({len(action_list_output)} rows)')

    # 3. Feature importance
    importance_df.to_sql('sqlsal_feature_importance', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Feature importance saved to {INPUT_SCHEMA}.sqlsal_feature_importance ({len(importance_df)} rows)')

    # 4. SHAP importance
    if len(shap_importance) > 0:
        shap_importance.to_sql('sqlsal_shap_importance', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
        logging.info(f'SHAP importance saved to {INPUT_SCHEMA}.sqlsal_shap_importance ({len(shap_importance)} rows)')

    # 5. Model metadata
    model_metadata = pd.DataFrame([{
        'model_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_filename': model_filename,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'train_roc_auc': roc_auc_score(y_train, y_proba_train),
        'test_roc_auc': roc_auc_score(y_test, y_proba_test),
        'best_optuna_score': best_score,
        'n_optuna_trials': N_OPTUNA_TRIALS,
        'probability_threshold': PROBABILITY_THRESHOLD,
        'high_prob_contacts': len(action_list),
        'training_date': datetime.now(),
        'best_params': str(best_params)
    }])
    model_metadata.to_sql('sqlsal_model_metadata', engine, schema=INPUT_SCHEMA, if_exists='append', index=False)
    logging.info(f'Model metadata saved to {INPUT_SCHEMA}.sqlsal_model_metadata')

    logging.info('=' * 70)
    logging.info('SQLSAL PIPELINE COMPLETED SUCCESSFULLY')
    logging.info('=' * 70)
    logging.info(f'Training samples: {len(X_train)}')
    logging.info(f'Test samples: {len(X_test)}')
    logging.info(f'Features: {len(X.columns)}')
    logging.info(f'Train ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}')
    logging.info(f'Test ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}')
    logging.info(f'Best Optuna ROC-AUC: {best_score:.4f}')
    logging.info(f'High-probability SQL contacts: {len(action_list)}')
    logging.info(f'Model saved: {model_filename}')
    logging.info('=' * 70)
