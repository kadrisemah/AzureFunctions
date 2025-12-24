import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, Table, Column, MetaData, text
from sqlalchemy.types import Integer, Float, String, DateTime, Boolean, NVARCHAR
from urllib.parse import quote_plus
from datetime import datetime
from sklearn.cluster import KMeans
import warnings
import os
import azure.functions as func

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def infer_sqlalchemy_type(dtype, column_name=None, series=None):
    """Infer SQLAlchemy type from pandas dtype"""
    dtype_str = str(dtype)
    if 'datetime' in dtype_str:
        return DateTime
    if dtype_str.startswith('int'):
        return Integer
    if dtype_str.startswith('float'):
        return Float
    if dtype_str == 'bool':
        return Boolean
    if dtype_str == 'object':
        long_text_columns = ['meetingnotesaggregated', 'callnotesaggregated', 'meetingnotespreview', 'callnotespreview', 'emailbodypreview', 'maildata', 'client_summary', 'client_empathy_map', 'psychology_profile']
        if column_name and any(keyword in column_name.lower() for keyword in long_text_columns):
            return NVARCHAR(None)
        if series is not None:
            max_len = series.astype(str).str.len().max()
            if pd.isna(max_len) or max_len == 0:
                return NVARCHAR(255)
            elif max_len > 4000:
                return NVARCHAR(None)
            elif max_len > 500:
                return NVARCHAR(4000)
            else:
                return NVARCHAR(min(max(int(max_len * 1.5), 50), 500))
        return NVARCHAR(500)
    return NVARCHAR(500)


def create_table_dynamically(df, engine, schema, table_name):
    """Create SQL table dynamically based on DataFrame schema"""
    metadata = MetaData()
    columns = []
    for col in df.columns:
        col_type = infer_sqlalchemy_type(df[col].dtype, col, df[col])
        columns.append(Column(col, col_type, nullable=True))

    table = Table(table_name, metadata, *columns, schema=schema)
    inspector = inspect(engine)

    if inspector.has_table(table_name, schema=schema):
        with engine.connect() as conn:
            conn.execute(text(f'DROP TABLE {schema}.{table_name}'))
            conn.commit()
            logging.info(f'Dropped existing table {schema}.{table_name}')

    metadata.create_all(engine)
    logging.info(f'Created table {schema}.{table_name}')
    return table


def save_to_sql_database(df, engine, schema, table_name):
    """Save DataFrame to SQL database with chunking"""
    # Clean column names
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('&', 'and') for col in df.columns]

    # Handle datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)

    # Handle text columns
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str).replace('nan', '')

    # Create table
    create_table_dynamically(df, engine, schema, table_name)

    # Insert in chunks
    chunk_size = 1000
    total_chunks = (len(df) - 1) // chunk_size + 1

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk.to_sql(name=table_name, con=engine, schema=schema, if_exists='append', index=False, method=None)
        logging.info(f'Saved chunk {i//chunk_size + 1}/{total_chunks} ({len(chunk)} rows)')


def perform_monthly_clustering(df, n_clusters=4):
    """Perform monthly KMeans clustering to create SegmentClass feature"""
    logging.info(f'Performing monthly KMeans clustering (k={n_clusters})...')

    # Select numeric features for clustering
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Client_ID', 'HubSpotId', 'OnboardingId', 'SegmentClass']
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    # Perform clustering
    if len(numeric_features) > 0:
        X_cluster = df[numeric_features].fillna(0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['SegmentClass'] = kmeans.fit_predict(X_cluster)

        # One-hot encode SegmentClass
        for i in range(n_clusters):
            df[f'SegmentClass_{i}'] = (df['SegmentClass'] == i).astype(int)

        logging.info(f'Clustering complete: {n_clusters} segments created')
        logging.info(f'Segment distribution: {df["SegmentClass"].value_counts().to_dict()}')
    else:
        logging.warning('No numeric features available for clustering')
        df['SegmentClass'] = 0
        for i in range(n_clusters):
            df[f'SegmentClass_{i}'] = (i == 0)

    return df


def main(mytimer: func.TimerRequest) -> None:
    logging.info('='*70)
    logging.info('TOPUP DATA PREPARATION FUNCTION STARTED')
    logging.info('='*70)

    # Configuration
    OUTPUT_SCHEMA = 'dbo'
    OUTPUT_TABLE = 'topup_training_dataset'

    logging.info('Retrieving credentials from application settings...')

    # PrimeAI Database (Enriched features)
    primeai_server = os.environ.get('PRIMEAI_SERVER')
    primeai_database = os.environ.get('PRIMEAI_DATABASE')
    primeai_user = os.environ.get('PRIMEAI_USER')
    primeai_password = os.environ.get('PRIMEAI_PASSWORD')

    # Target Database
    target_server = os.environ.get('TARGET_SERVER')
    target_database = os.environ.get('TARGET_DATABASE')
    target_user = os.environ.get('TARGET_USER')
    target_password = os.environ.get('TARGET_PASSWORD')

    # Validate credentials
    if not all([primeai_server, primeai_database, primeai_user, primeai_password,
                target_server, target_database, target_user, target_password]):
        logging.error('Missing required environment variables')
        raise ValueError('Missing required environment variables. Need: PRIMEAI_SERVER, PRIMEAI_DATABASE, PRIMEAI_USER, PRIMEAI_PASSWORD, TARGET_SERVER, TARGET_DATABASE, TARGET_USER, TARGET_PASSWORD')

    logging.info('Credentials retrieved successfully')

    # Create database connections
    logging.info('Establishing database connections...')

    driver = 'ODBC Driver 18 for SQL Server'

    # PrimeAI connection
    primeai_connection_string = (
        f'mssql+pyodbc://{primeai_user}:{quote_plus(primeai_password)}@{primeai_server}:1433/{primeai_database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    primeai_engine = create_engine(primeai_connection_string)
    logging.info(f'Connected to PrimeAI database: {primeai_database}')

    # Target connection
    target_connection_string = (
        f'mssql+pyodbc://{target_user}:{quote_plus(target_password)}@{target_server}:1433/{target_database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    target_engine = create_engine(target_connection_string)
    logging.info(f'Connected to target database: {target_database}')

    # Query to fetch ALL features from PrimeAiMandates
    logging.info('Fetching enriched features from PrimeAiMandates...')

    primeai_query = """
    SELECT
          [Client ID]
        ,[ContactType]
        ,[LeadStageNew]
        ,[Nationality]
        ,[SegmentClass]
        ,[DealCloseDateCommittedCapitalContact]
        ,[wealth_range]
        ,[engagement_label]
        ,[client_approached_us]
        ,[share_of_wallet_label]
        ,[digital_campaign_influence]
        ,[MaritalStatus]
        ,[EmploymentField]
        ,[ShariaCompliant]
        ,[AgeRange]
        ,[TypeOfReferral]
        ,[Gender]
        ,[TotalWebinarsAttended]
        ,[TotalWebinarsRegistered]
        ,[MarketingEmailsClicked]
        ,[MarketingEmailsOpened]
        ,[MarketingEmailsDelivered]
        ,[TotalCallsConnected]
        ,[TotalCallsMade]
        ,[meetscounts]
        ,[MEETINGNOTESAGGREGATED]
        ,[CALLNOTESAGGREGATED]
        ,[callscountsconnected]
        ,[opened_counts]
        ,[clicked_counts]
        ,[delivered_counts]
        ,[maildata]
        ,[ContactType_DFO]
        ,[ContactType_TFO]
        ,[FORMS_FILLED]
        ,[numberofformsfilled]
        ,[WEBSITEVISTED]
        ,[numberofwebvisits]
        ,[CLIENTAPP]
        ,[numberofclientappvisits]
        ,[PJVISITED]
        ,[PJVISITS]
        ,[Total Top Ups]
        ,[Total Top Up Amount]
        ,[total_commitment]
        ,[AssetClass_Cash & Cash Equivalent]
        ,[AssetClass_Credit]
        ,[AssetClass_Equities]
        ,[AssetClass_Multi-Asset Class]
        ,[AssetClass_Others]
        ,[AssetClass_Private Equity]
        ,[AssetClass_Real Estate]
        ,[AssetClass_Yielding Investments]
        ,[SubStrategy_Absolute Return]
        ,[SubStrategy_Capital Growth]
        ,[SubStrategy_Capital Yielding]
        ,[SubStrategy_Opportunistic]
        ,[SubStrategy_Others]
        ,[First Commitment Amount]
        ,[DEALCLOSEDATECOMMITTEDCAPITAL]
        ,[Top Up Frequency]
        ,[they_topped_up]
        ,[Activities per Quarter]
        ,[Hubspot_Persona]
        ,[SegmentClass]
        ,[RM_Name]
        ,[Client Summary]
        ,[Client Empathy Map]
        ,[Psychology Profile]
    FROM [dbo].[PrimeAiMandates]
    """

    mandate_pipeline = pd.read_sql_query(primeai_query, primeai_engine)
    logging.info(f'Fetched {len(mandate_pipeline)} rows from PrimeAiMandates')
    logging.info(f'Columns: {len(mandate_pipeline.columns)} features')

    # Perform monthly clustering to update SegmentClass
    mandate_pipeline = perform_monthly_clustering(mandate_pipeline, n_clusters=4)

    # Add metadata
    mandate_pipeline['created_at'] = datetime.now()
    mandate_pipeline['pipeline_run_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.info(f'Final dataset: {len(mandate_pipeline)} rows, {len(mandate_pipeline.columns)} columns')

    # Save to target database
    logging.info(f'Saving training dataset to {OUTPUT_SCHEMA}.{OUTPUT_TABLE}...')
    save_to_sql_database(mandate_pipeline, target_engine, OUTPUT_SCHEMA, OUTPUT_TABLE)
    logging.info(f'Successfully saved {len(mandate_pipeline)} rows to {OUTPUT_SCHEMA}.{OUTPUT_TABLE}')

    logging.info('='*70)
    logging.info('TOPUP DATA PREPARATION COMPLETED SUCCESSFULLY')
    logging.info('='*70)
