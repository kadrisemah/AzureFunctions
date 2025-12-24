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
import requests
from tqdm import tqdm
import azure.functions as func

warnings.filterwarnings('ignore')


def create_db_connection(server, database, username, password):
    driver = 'ODBC Driver 18 for SQL Server'
    connection_string = (
        f'mssql+pyodbc://{username}:{quote_plus(password)}@{server}:1433/{database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    return create_engine(connection_string)


def explain_catboost_client_savefig(
    hid,
    X_test,
    model,
    categorical_feats,
    n_display=10,
    max_label_len=25,
    max_val_len=30
):
    """
    Generate SHAP explanation for a single client
    Returns DataFrame with Feature, Value, SHAP columns
    """
    if hid not in X_test.index:
        logging.warning(f"{hid} not in X_test")
        return "_"

    row = X_test.loc[[hid]].loc[:, ~X_test.columns.duplicated()]
    cat_features_idx = [row.columns.get_loc(c) for c in categorical_feats if c in row.columns]
    pool = Pool(row, cat_features=cat_features_idx)
    shap_vec = model.get_feature_importance(pool, type="ShapValues")[0]
    shap_feat = shap_vec[:-1]
    base_val = shap_vec[-1]
    feature_vals = row.values[0]
    feature_names = list(row.columns)
    short_names = [name if len(name) <= max_label_len else name[:max_label_len - 1] + "…" for name in feature_names]
    display_vals = [v[:max_val_len-1] + "…" if isinstance(v, str) and len(v) > max_val_len else v for v in feature_vals]

    shap_values = shap_vec[:-1]
    df = pd.DataFrame({
        'Feature': feature_names,
        'Value': display_vals,
        'SHAP': shap_values
    }).sort_values(by='SHAP', key=abs, ascending=False).reset_index(drop=True)
    return df


def generate_pitch_with_shap(client_summary, topup_probability, shap_explanation, azure_api_url, azure_api_key):
    """
    Generate AI pitch using Azure OpenAI GPT-4
    """
    pitch_prompt = f"""
    You are a Senior Financial Advisor at a prestigious family office. Your role is to clearly and persuasively explain AI-driven top-up predictions to a Relationship Manager (RM) who works directly with clients.

    Please:
    - Analyze the attached SHAP summary , which shows the key variables driving the AI model's probability that the client will make a top-up.
    - Explain, in plain business language, how each **detected** variable (ignore variables that are missing or 'NaN') contributes to the model's prediction.
    - Link your explanation of variable effects directly to the model's  TOPUP **probability** (provided below).
    - For each variable in the SHAP plot:
        - Briefly explain what the variable means in business terms.
        - Describe if the variable is **increasing** or **decreasing** the probability, and why.
        - Do NOT mention or speculate about variables that are missing or have 'NaN' values.
    - Make your explanation practical: focus on what a Relationship Manager should understand or tell the client.
    - Close with a short summary: explain why the probability is what it is, in light of the variables detected.
    If a name exists in the data ommit it and only say the client
    Remember: Do not use technical jargon or statistical terms. Assume the RM has no technical background but needs actionable business insights.
    Do not put values from the the shap as numerical just interpret how they might have affected the result in a smart way
    please make it smart as if you are addressing to relationship manager so dont add meaningless phrases like based on this data or thank you for the data just put you best analysis

    """

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": pitch_prompt},
                    {"type": "text", "text": f"Client Summary: {client_summary}"},
                    {"type": "text", "text": f"AI Model Probability to Top-up: {topup_probability:.2f}"},
                    {"type": "text", "text": f"Shap explanation: {shap_explanation}"},
                ]
            }
        ],
        "max_tokens": 500
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": azure_api_key
    }

    try:
        response = requests.post(azure_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error generating pitch: {e}")
        return ""


def main(mytimer: func.TimerRequest) -> None:
    logging.info('='*70)
    logging.info('TOPUP MODEL TRAINING FUNCTION STARTED')
    logging.info('='*70)

    # Configuration
    INPUT_SCHEMA = 'dbo'
    INPUT_TABLE = 'topup_training_dataset'
    TRAIN_CUTOFF = '2024-06-01'
    TEST_MONTH = '2025-07-01'
    PROBABILITY_THRESHOLD = 0.5

    logging.info('Retrieving credentials from application settings...')

    # Target Database (where training data and predictions are saved)
    db_server = os.environ.get('TARGET_SERVER')
    db_database = os.environ.get('TARGET_DATABASE')
    db_username = os.environ.get('TARGET_USER')
    db_password = os.environ.get('TARGET_PASSWORD')

    # PrimeAI Database (to fetch Client Summary, Empathy Map, Psychology Profile)
    primeai_server = os.environ.get('PRIMEAI_SERVER')
    primeai_database = os.environ.get('PRIMEAI_DATABASE')
    primeai_user = os.environ.get('PRIMEAI_USER')
    primeai_password = os.environ.get('PRIMEAI_PASSWORD')

    # Azure OpenAI
    azure_api_url = os.environ.get('AZURE_OPENAI_ENDPOINT')
    azure_api_key = os.environ.get('AZURE_OPENAI_KEY')

    if not all([db_server, db_database, db_username, db_password,
                primeai_server, primeai_database, primeai_user, primeai_password,
                azure_api_url, azure_api_key]):
        logging.error('Missing required environment variables')
        raise ValueError('Missing required environment variables. Need: TARGET_*, PRIMEAI_*, AZURE_OPENAI_*')

    logging.info('Credentials retrieved successfully')

    logging.info('Establishing database connections...')
    engine = create_db_connection(db_server, db_database, db_username, db_password)
    primeai_engine = create_db_connection(primeai_server, primeai_database, primeai_user, primeai_password)
    logging.info('Database connections established')

    logging.info(f'Loading training data from {INPUT_SCHEMA}.{INPUT_TABLE}...')
    query = f'SELECT * FROM {INPUT_SCHEMA}.{INPUT_TABLE}'
    data = pd.read_sql(query, engine)
    logging.info(f'Loaded {len(data)} rows, {len(data.columns)} columns')

    # Rename Client ID to HubSpotId if needed
    if 'Client ID' in data.columns and 'HubSpotId' not in data.columns:
        data.rename(columns={'Client ID': 'HubSpotId'}, inplace=True)

    logging.info(f'Unique clients: {data["HubSpotId"].nunique()}')

    # Load PrimeAI metadata for output enrichment
    logging.info('Loading client metadata from PrimeAI...')
    primeai_query = """
    SELECT
        [Client ID] as HubSpotId,
        [Top Up Frequency],
        [they_topped_up],
        [Activities per Quarter],
        [Hubspot_Persona],
        [RM_Name],
        [Client Summary],
        [Client Empathy Map],
        [Psychology Profile]
    FROM [dbo].[PrimeAiMandates]
    """
    primeai_metadata = pd.read_sql(primeai_query, primeai_engine)
    logging.info(f'Loaded metadata for {len(primeai_metadata)} clients')

    logging.info('Preparing features...')
    exclude_cols = [
        'HubSpotId', 'Client ID', 'TOPUP', 'TOPUPAMOUNT',
        'created_at', 'pipeline_run_id',
        'Client Summary', 'Client Empathy Map', 'Psychology Profile',
        'RM_Name', 'Top Up Frequency', 'they_topped_up', 'Activities per Quarter',
        'Hubspot_Persona', 'MEETINGNOTESAGGREGATED', 'CALLNOTESAGGREGATED',
        'meetingnotesaggregated', 'callnotesaggregated', 'meetingnotespreview',
        'callnotespreview', 'emailbodypreview', 'maildata', 'client_summary',
        'client_empathy_map', 'psychology_profile'
    ]

    # Create binary target
    logging.info('Creating target variable...')
    data['target_flag'] = (data['they_topped_up'].fillna(0) > 0).astype(int)
    target_dist = data['target_flag'].value_counts()
    logging.info(f'Target distribution: {dict(target_dist)}')
    logging.info(f'Positive rate: {data["target_flag"].mean():.2%}')

    # Select numeric features
    numeric_cols = data.select_dtypes(include=['number']).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and c != 'target_flag']

    # Encode categorical features
    categorical_cols = ['ContactType', 'LeadStageNew', 'Nationality', 'MaritalStatus',
                       'EmploymentField', 'ShariaCompliant', 'AgeRange', 'TypeOfReferral',
                       'Gender', 'wealth_range', 'engagement_label', 'client_approached_us',
                       'share_of_wallet_label', 'digital_campaign_influence']
    categorical_cols = [c for c in categorical_cols if c in data.columns]

    for col in categorical_cols:
        if col not in feature_cols:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown').astype(str))
            feature_cols.append(f'{col}_encoded')

    logging.info(f'Selected {len(feature_cols)} features')

    # Outlier detection
    logging.info('Detecting outliers...')
    contamination = 0.05
    X_outlier = data[feature_cols].fillna(0)
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_labels = iso_forest.fit_predict(X_outlier)
    data['is_outlier'] = (outlier_labels == -1).astype(int)
    n_outliers = data['is_outlier'].sum()
    logging.info(f'Detected {n_outliers} outliers ({n_outliers/len(data):.2%})')

    # Train/test split
    logging.info(f'Splitting data: train <= {TRAIN_CUTOFF}...')
    train_df = data.copy()

    # Remove outliers from training
    if 'is_outlier' in train_df.columns:
        original_train_size = len(train_df)
        train_df = train_df[train_df['is_outlier'] == 0]
        logging.info(f'Removed {original_train_size - len(train_df)} outliers from training')

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target_flag']

    # For predictions, we use ALL data (training set only had historical data)
    X_all = data[feature_cols].fillna(0)
    X_all.index = data['HubSpotId'].values

    logging.info(f'Train: {len(X_train)} samples, {len(X_train.columns)} features')
    logging.info(f'Train positive rate: {y_train.mean():.2%}')

    # Identify categorical feature indices
    cat_features = [i for i, c in enumerate(X_train.columns) if c.endswith('_encoded')]
    categorical_feats = [c for c in X_train.columns if c.endswith('_encoded')]
    logging.info(f'Categorical features: {len(cat_features)}')

    # Train CatBoost model
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
        class_weights={0: 1.0, 1: 0.95},
        od_type='Iter',
        early_stopping_rounds=100,
        verbose=100
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    model.fit(train_pool, use_best_model=True)
    logging.info('Model training completed')

    # Evaluate on training set
    logging.info('Evaluating model...')
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]

    logging.info('TRAIN SET PERFORMANCE')
    logging.info(classification_report(y_train, y_pred_train))
    logging.info(f'ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}')

    # Generate predictions for ALL clients
    logging.info('Generating predictions for all clients...')
    y_proba_all = model.predict_proba(X_all)[:, 1]

    predictions_df = pd.DataFrame({
        'HubSpotId': X_all.index,
        'TOPUP_PROBABILITY_THIS_MONTH': y_proba_all
    })

    # Merge with PrimeAI metadata
    predictions_df = predictions_df.merge(primeai_metadata, on='HubSpotId', how='left')

    logging.info('Analyzing feature importance...')
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logging.info('Top 20 Feature Importances:')
    logging.info(importance_df.head(20).to_string(index=False))

    # Save model
    model_filename = f'/tmp/catboost_topup_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.cbm'
    model.save_model(model_filename)
    logging.info(f'Model saved: {model_filename}')

    # Generate action list (high-probability clients)
    logging.info(f'Generating action list (threshold={PROBABILITY_THRESHOLD})...')
    action_list = predictions_df[predictions_df['TOPUP_PROBABILITY_THIS_MONTH'] >= PROBABILITY_THRESHOLD].copy()
    action_list = action_list.sort_values('TOPUP_PROBABILITY_THIS_MONTH', ascending=False)
    logging.info(f'Action list: {len(action_list)} high-probability clients')

    if len(action_list) > 0:
        logging.info(f'Average probability: {action_list["TOPUP_PROBABILITY_THIS_MONTH"].mean():.2%}')
        logging.info('Top 10 Clients:')
        logging.info(str(action_list.head(10)[['HubSpotId', 'TOPUP_PROBABILITY_THIS_MONTH']]))

        # Generate SHAP explanations and AI pitches for high-probability clients
        logging.info('Generating SHAP explanations and AI pitches for high-probability clients...')
        explainability = {}
        pitches = {}
        errors_empty = []

        for idx, row in tqdm(action_list.iterrows(), total=len(action_list), desc="Generating explanations"):
            hid = int(row['HubSpotId'])

            # Generate SHAP explanation
            shap_df = explain_catboost_client_savefig(
                hid,
                X_all,
                model,
                categorical_feats
            )

            if isinstance(shap_df, str) and shap_df == '_':
                explainability[idx] = ""
                pitches[idx] = ""
                errors_empty.append(hid)
            else:
                try:
                    client_summary = str(row.get('Client Summary', ''))
                    probability = row['TOPUP_PROBABILITY_THIS_MONTH']
                    explained_vals = str(shap_df)

                    # Generate AI pitch
                    pitch = generate_pitch_with_shap(
                        client_summary,
                        probability,
                        explained_vals,
                        azure_api_url,
                        azure_api_key
                    )

                    explainability[idx] = explained_vals
                    pitches[idx] = pitch
                except Exception as e:
                    logging.error(f"Error processing client {hid}: {e}")
                    explainability[idx] = ""
                    pitches[idx] = ""
                    errors_empty.append(hid)

        # Add explanations and pitches to action list
        action_list['explain'] = action_list.index.map(lambda x: explainability.get(x, ""))
        action_list['pitch'] = action_list.index.map(lambda x: pitches.get(x, ""))

        logging.info(f'Generated {len([v for v in pitches.values() if v])} AI pitches')
        logging.info(f'Errors/empty: {len(errors_empty)} clients')

    # Save results to SQL
    logging.info('Saving results to SQL...')

    # Save all predictions
    predictions_df.to_sql('topup_model_predictions', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Predictions saved to {INPUT_SCHEMA}.topup_model_predictions ({len(predictions_df)} rows)')

    # Save action list with explanations and pitches
    if len(action_list) > 0:
        action_list.to_sql('topup_rm_action_list', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
        logging.info(f'Action list saved to {INPUT_SCHEMA}.topup_rm_action_list ({len(action_list)} rows)')

    # Save feature importance
    importance_df.to_sql('topup_feature_importance', engine, schema=INPUT_SCHEMA, if_exists='replace', index=False)
    logging.info(f'Feature importance saved to {INPUT_SCHEMA}.topup_feature_importance')

    logging.info('=' * 70)
    logging.info('TOPUP MODEL TRAINING COMPLETED SUCCESSFULLY')
    logging.info('=' * 70)
    logging.info(f'Training samples: {len(X_train)}')
    logging.info(f'Total predictions: {len(predictions_df)}')
    logging.info(f'Features: {len(X_train.columns)}')
    logging.info(f'Train ROC-AUC: {roc_auc_score(y_train, y_proba_train):.4f}')
    logging.info(f'High-probability clients: {len(action_list)}')
    logging.info(f'Model saved: {model_filename}')
    logging.info('=' * 70)
