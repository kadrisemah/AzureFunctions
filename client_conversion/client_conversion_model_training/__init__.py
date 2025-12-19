import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import pyodbc
import warnings
import os
import azure.functions as func

warnings.filterwarnings('ignore')


def _pick_sql_server_driver():
    """Pick an installed Microsoft SQL Server ODBC driver (prefer 18, then 17)"""
    installed = [d.strip() for d in pyodbc.drivers()]
    for candidate in ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"]:
        if candidate in installed:
            return candidate
    raise RuntimeError(
        "No supported SQL Server ODBC driver found. "
        "Install 'ODBC Driver 18 for SQL Server' or 'ODBC Driver 17 for SQL Server'. "
        f"Installed drivers: {installed}"
    )


def create_db_connection(server, database, username, password):
    """Create database connection using ODBC Driver 18 or 17"""
    driver = _pick_sql_server_driver()

    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=30;"
    )

    odbc_connect = quote_plus(odbc_str)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={odbc_connect}", fast_executemany=True)
    return engine


def main(mytimer: func.TimerRequest) -> None:
    logging.info('Client Conversion Model Training Function started')

    INPUT_SCHEMA = 'dbo'
    INPUT_TABLE = 'client_conversion_training_dataset'
    N_OPTUNA_TRIALS = 50  # Adjust based on computational budget

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
    data_final = pd.read_sql(query, engine)
    logging.info(f'Loaded {len(data_final)} rows, {len(data_final.columns)} columns')

    # Prepare data for modeling
    logging.info('Preparing features for modeling...')

    # Columns to drop (non-features, target, metadata)
    columns_to_drop = [
        "GAA23PriorityPassRequested", 'LeadScoreProspectJourney', "GAA23PriorityPass",
        'EstimatedClientSegment', 'Team', 'Flow', 'HashID', 'CreateDay',
        'ContactOwnerId', "BirthYear", "ExpectedClosingProbability", 'SourceExtractionDate',
        'HsPersona', 'SegmentCount_0', 'ProspectJourneyQualificationStatus',
        'SourceOfFundsConsolidated', 'CloseDate', 'HS_DaysToClose',
        'DealCloseDateCommittedCapitalContact', 'DAYSTOCLOSE', 'WeekNumber',
        'YearNumber', 'LeadStageNewCEO', 'LeadStageNew', 'TotalDealAmount',
        "ExistingMappedLeadStages", 'LeadStage', 'OnboardingId', 'TotalDealAmountDS',
        "ExpectedMonthOfclosing", "GAA23ListMembership", "GAA23InviteeType", "GAA23EntryDate",
        'HSLeadStatus', 'IPCountry', 'BirthDate', 'InterestedFlow',
        "PropensityScoreDFOSALs", "GAA23EntryTime", "GAA23CheckedInBy", "IPCity"
    ]

    # Extract target before dropping
    if 'Target' not in data_final.columns:
        logging.error('Target column not found in dataset')
        raise ValueError('Target column missing')

    y = data_final['Target'].copy()

    # Drop non-feature columns
    existing_drop_cols = [col for col in columns_to_drop if col in data_final.columns]
    X = data_final.drop(columns=existing_drop_cols + ['Target'], errors='ignore')

    # Fill missing values
    X = X.fillna(0)

    # Identify categorical features
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    logging.info(f'Categorical features: {len(categorical_features)}')

    for col in categorical_features:
        X[col] = X[col].astype(str)

    logging.info(f'Total dataset size: {len(X)}')
    logging.info(f'Features: {X.shape[1]}')
    logging.info(f'Class distribution:\n{y.value_counts().to_dict()}')

    # Optuna objective function with out-of-fold validation
    def objective(trial):
        """
        Optuna objective function using out-of-fold validation
        """

        # Hyperparameter search space (EXACTLY as in notebook)
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),

            # Fixed parameters
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'cat_features': categorical_features,
            'task_type': 'CPU',
            'early_stopping_rounds': 50
        }

        # 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            cv_scores.append(score)

        return np.mean(cv_scores)

    # Run Optuna optimization
    logging.info('='*70)
    logging.info(f'Starting Optuna Hyperparameter Optimization ({N_OPTUNA_TRIALS} trials)...')
    logging.info('='*70)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False,
        n_jobs=1  # Sequential for Azure Functions
    )

    best_params = study.best_params.copy()
    best_score = study.best_value

    logging.info('='*70)
    logging.info('Optimization Complete!')
    logging.info('='*70)
    logging.info(f'Best AUC Score: {best_score:.4f}')
    logging.info(f'Best Hyperparameters: {best_params}')

    # Prepare best parameters for final model
    best_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'cat_features': categorical_features,
        'task_type': 'CPU',
        'early_stopping_rounds': 50
    })

    # Generate out-of-fold predictions for ALL data
    logging.info('='*70)
    logging.info('Generating Out-of-Fold Predictions for ALL Data...')
    logging.info('='*70)

    oof_predictions = np.zeros(len(X))
    oof_predictions_proba = np.zeros(len(X))
    models = []  # Store all fold models

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f'Training Fold {fold}/5...')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model on this fold
        model = CatBoostClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        # Predict on validation fold (unseen data for this model)
        oof_predictions[val_idx] = model.predict(X_val)
        oof_predictions_proba[val_idx] = model.predict_proba(X_val)[:, 1]

        # Store model
        models.append(model)

        # Fold metrics
        fold_auc = roc_auc_score(y_val, oof_predictions_proba[val_idx])
        logging.info(f'Fold {fold} AUC: {fold_auc:.4f}')

    # Overall OOF evaluation
    logging.info('='*70)
    logging.info('OUT-OF-FOLD EVALUATION (ALL DATA)')
    logging.info('='*70)

    logging.info('Classification Report:')
    report = classification_report(y, oof_predictions, digits=4)
    logging.info(f'\n{report}')

    cm = confusion_matrix(y, oof_predictions)
    logging.info(f'Confusion Matrix:\n{cm}')

    logging.info('\nDetailed Metrics:')
    logging.info(f'Accuracy:  {accuracy_score(y, oof_predictions):.4f}')
    logging.info(f'Precision: {precision_score(y, oof_predictions):.4f}')
    logging.info(f'Recall:    {recall_score(y, oof_predictions):.4f}')
    logging.info(f'F1-Score:  {f1_score(y, oof_predictions):.4f}')
    logging.info(f'ROC-AUC:   {roc_auc_score(y, oof_predictions_proba):.4f}')

    # Train final model on ALL data for production
    logging.info('='*70)
    logging.info('Training Final Model on ALL Data for Production...')
    logging.info('='*70)

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y, verbose=False)

    # Save model
    model_filename = f'/tmp/catboost_client_conversion_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.cbm'
    final_model.save_model(model_filename)
    logging.info(f'Model saved: {model_filename}')

    # Generate predictions for all data
    logging.info('Generating predictions for all contacts...')
    y_proba_all = final_model.predict_proba(X)[:, 1]
    y_pred_all = final_model.predict(X)

    # Create predictions DataFrame
    predictions_df = data_final[['HubSpotId']].copy() if 'HubSpotId' in data_final.columns else pd.DataFrame()
    predictions_df['y_true'] = y.values
    predictions_df['y_pred'] = y_pred_all
    predictions_df['probability'] = y_proba_all
    predictions_df['oof_prediction'] = oof_predictions
    predictions_df['oof_probability'] = oof_predictions_proba
    predictions_df['prediction_date'] = datetime.now()

    # Feature importance (averaged across folds)
    logging.info('Analyzing feature importance...')
    feature_importance_list = []
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'fold': i+1
        })
        feature_importance_list.append(fold_importance)

    feature_importance_df = pd.concat(feature_importance_list)
    feature_importance_avg = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
    feature_importance_avg = feature_importance_avg.sort_values('importance', ascending=False)

    logging.info('Top 20 Feature Importances (Averaged Across Folds):')
    logging.info('\n' + feature_importance_avg.head(20).to_string(index=False))

    # Generate action list (high probability SAL prospects)
    logging.info('Generating action list for relationship managers...')
    PROBABILITY_THRESHOLD = 0.5

    action_list = predictions_df[predictions_df['probability'] >= PROBABILITY_THRESHOLD].sort_values('probability', ascending=False).copy()

    logging.info(f'Action list: {len(action_list)} high-probability SAL prospects')
    if len(action_list) > 0:
        logging.info(f'Average probability: {action_list["probability"].mean():.2%}')
        logging.info(f'Probability range: {action_list["probability"].min():.2%} - {action_list["probability"].max():.2%}')

    # Save results to SQL
    logging.info('Saving results to SQL...')

    # 1. All predictions (OOF)
    predictions_output = predictions_df[[col for col in ['HubSpotId', 'y_true', 'y_pred', 'probability', 'oof_prediction', 'oof_probability', 'prediction_date'] if col in predictions_df.columns]].copy()
    predictions_output.to_sql('client_conversion_model_predictions', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Predictions saved to {INPUT_SCHEMA}.client_conversion_model_predictions ({len(predictions_output)} rows)')

    # 2. Action list for sales team
    if 'HubSpotId' in action_list.columns:
        action_list_output = action_list[['HubSpotId', 'probability', 'oof_probability']].copy()
    else:
        action_list_output = action_list[['probability', 'oof_probability']].copy()

    action_list_output.to_sql('client_conversion_rm_action_list', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Action list saved to {INPUT_SCHEMA}.client_conversion_rm_action_list ({len(action_list_output)} rows)')

    # 3. Feature importance (averaged)
    feature_importance_avg.to_sql('client_conversion_feature_importance', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Feature importance saved to {INPUT_SCHEMA}.client_conversion_feature_importance ({len(feature_importance_avg)} rows)')

    # 4. Model metadata
    model_metadata = pd.DataFrame([{
        'model_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'model_filename': model_filename,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'oof_roc_auc': roc_auc_score(y, oof_predictions_proba),
        'oof_accuracy': accuracy_score(y, oof_predictions),
        'oof_f1_score': f1_score(y, oof_predictions),
        'best_optuna_score': best_score,
        'n_optuna_trials': N_OPTUNA_TRIALS,
        'probability_threshold': PROBABILITY_THRESHOLD,
        'high_prob_contacts': len(action_list),
        'training_date': datetime.now(),
        'best_params': str(best_params)
    }])
    model_metadata.to_sql('client_conversion_model_metadata', engine, schema=INPUT_SCHEMA, if_exists='append', index=False)
    logging.info(f'Model metadata saved to {INPUT_SCHEMA}.client_conversion_model_metadata')

    # 5. Optuna study results
    study_df = study.trials_dataframe()
    study_df.to_sql('client_conversion_optuna_study', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Optuna study results saved to {INPUT_SCHEMA}.client_conversion_optuna_study')

    logging.info('='*70)
    logging.info('CLIENT CONVERSION PIPELINE COMPLETED SUCCESSFULLY')
    logging.info('='*70)
    logging.info(f'Total samples: {len(X)}')
    logging.info(f'Features: {len(X.columns)}')
    logging.info(f'OOF ROC-AUC: {roc_auc_score(y, oof_predictions_proba):.4f}')
    logging.info(f'OOF Accuracy: {accuracy_score(y, oof_predictions):.4f}')
    logging.info(f'Best Optuna ROC-AUC: {best_score:.4f}')
    logging.info(f'High-probability SAL prospects: {len(action_list)}')
    logging.info(f'Model saved: {model_filename}')
    logging.info('='*70)
