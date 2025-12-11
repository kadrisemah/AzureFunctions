import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, Table, Column, MetaData, text
from sqlalchemy.types import Integer, Float, String, DateTime, Boolean, NVARCHAR
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from itertools import product
import warnings
import os
import azure.functions as func

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def infer_sqlalchemy_type(dtype, column_name=None, series=None):
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
        long_text_columns = ['meetingnotespreview', 'callnotespreview', 'emailbodypreview', 'sales_email', 'marketing_email']
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
    metadata.create_all(engine)
    return table


def save_to_sql_database(df, engine, schema, table_name):
    df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str).replace('nan', '')
    create_table_dynamically(df, engine, schema, table_name)
    chunk_size = 1000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk.to_sql(name=table_name, con=engine, schema=schema, if_exists='append', index=False, method=None)


def create_db_connection(server, database, username, password):
    driver = 'ODBC Driver 18 for SQL Server'
    connection_string = (
        f'mssql+pyodbc://{username}:{quote_plus(password)}@{server}:1433/{database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    return create_engine(connection_string)


def main(mytimer: func.TimerRequest) -> None:
    logging.info('Data Preparation Function started')

    OUTPUT_SCHEMA = 'dbo'
    OUTPUT_TABLE = 'topup_training_dataset'

    logging.info('Retrieving credentials from application settings...')

    source_server = os.environ.get('SOURCE_SERVER')
    source_database = os.environ.get('SOURCE_DATABASE')
    source_username = os.environ.get('SOURCE_USER')
    source_password = os.environ.get('SOURCE_PASSWORD')

    target_server = os.environ.get('TARGET_SERVER')
    target_database = os.environ.get('TARGET_DATABASE')
    target_username = os.environ.get('TARGET_USER')
    target_password = os.environ.get('TARGET_PASSWORD')

    if not all([source_server, source_database, source_username, source_password,
                target_server, target_database, target_username, target_password]):
        logging.error('Missing required environment variables. Please configure all database credentials in Application Settings.')
        raise ValueError('Missing required environment variables')

    logging.info('Credentials retrieved successfully')

    logging.info('Establishing database connections...')
    source_engine = create_db_connection(source_server, source_database, source_username, source_password)
    target_engine = create_db_connection(target_server, target_database, target_username, target_password)
    logging.info('Database connections established')

    logging.info('Extracting client IDs...')
    Client_ID_Data_query = """SELECT DISTINCT associatedcontactid
FROM (
    SELECT associatedcontactid, Deal_Stage, SUM(amount) AS total_amount
    FROM vi_HSDeal
    WHERE Deal_Stage LIKE '%committed%'
    GROUP BY associatedcontactid, Deal_Stage
    HAVING SUM(amount) > 0
) AS filtered_deals;
"""
    df_client_IDs = pd.read_sql(Client_ID_Data_query, source_engine)
    df_client_IDs = df_client_IDs[df_client_IDs['associatedcontactid'].notna()].reset_index(drop=True)
    logging.info(f'Client IDs extracted: {len(df_client_IDs)} records')

    logging.info('Extracting contacts and deals...')
    Contact_Data_query = """SELECT *
FROM vi_HSContact
WHERE leadstagenew LIKE '%Committed%'
  AND (IsThisLeadAlreadyFoundInAnInternalDatabase IS NULL OR IsThisLeadAlreadyFoundInAnInternalDatabase NOT IN ('Internal', 'Tester'));
"""
    df_contact = pd.read_sql(Contact_Data_query, source_engine)
    df_deal = pd.read_sql('SELECT * FROM vi_HSDeal', source_engine)
    logging.info(f'Contacts: {len(df_contact)}, Deals: {len(df_deal)}')

    logging.info('Extracting engagement data...')
    df_engagement = pd.read_sql('SELECT * FROM vi_HSEngagementMeeting;', source_engine)
    OG_df_calls = pd.read_sql('SELECT * FROM vi_HSEngagementCall;', source_engine)
    OG_df_sales_email = pd.read_sql('SELECT AssociatedContactId, CreateDate, EmailBodyPreview, EmailDirection FROM vi_HSEngagementEmail;', source_engine)
    OG_df_marketing_email = pd.read_sql('SELECT createdAt, emailSubject FROM vi_HSMarketingEmail;', source_engine)
    logging.info(f'Meetings: {len(df_engagement)}, Calls: {len(OG_df_calls)}, Emails: {len(OG_df_sales_email)}')

    logging.info('Extracting risk profiles and contact owners...')
    OG_df_Risk_Profile = pd.read_sql('SELECT * FROM conform.PM1Risk;', source_engine)
    Contact_Owner = pd.read_sql('SELECT Ownerid, OwnerFullName, Email FROM vi_hsowner;', source_engine)
    logging.info(f'PM1 Risk: {len(OG_df_Risk_Profile)}, Owners: {len(Contact_Owner)}')

    logging.info('Identifying committed capital clients...')
    df_deal['Deal_Stage_clean'] = df_deal['Deal_Stage'].str.lower().str.strip()
    df_deal_filtered_ids = df_deal[
        df_deal['Deal_Stage_clean'].str.contains('committed capital', na=False) &
        df_deal['ASSOCIATEDCONTACTID'].notna() &
        (df_deal['AMOUNT'].astype(float) > 0)
    ]
    df_contact_filtered = df_contact[
        (df_contact['LeadStageNew'] == 'COMMITTED CAPITAL') &
        (df_contact['IsThisLeadAlreadyFoundInAnInternalDatabase'].isna() |
         ~df_contact['IsThisLeadAlreadyFoundInAnInternalDatabase'].isin(['Internal', 'Tester']))
    ]
    df_result = df_contact_filtered.merge(df_deal_filtered_ids, left_on='HubSpotId', right_on='ASSOCIATEDCONTACTID', how='inner')
    df_result['is_client'] = 1
    logging.info(f'Identified {df_result["HubSpotId"].nunique()} unique clients')

    logging.info('Merging contact owners...')
    df_result = df_result.merge(Contact_Owner[['Ownerid', 'OwnerFullName']], how='left', left_on='ContactOwnerId', right_on='Ownerid')
    df_result = df_result.drop(columns=['ContactOwnerId', 'Ownerid']).rename(columns={'OwnerFullName': 'ContactOwner'})
    list_id_clients = df_result.HubSpotId.unique()
    filtered_df_engagement = df_engagement[df_engagement['AssociatedContactId'].isin(list_id_clients)]
    logging.info(f'Filtered engagement: {len(filtered_df_engagement)} records')

    logging.info('Creating monthly panel dataset...')
    df_txn = df_result.copy().loc[:, ~df_result.columns.duplicated()]
    df_msg = filtered_df_engagement.copy().loc[:, ~filtered_df_engagement.columns.duplicated()]
    df_txn['CLOSEDATE'] = pd.to_datetime(df_txn['CLOSEDATE'], errors='coerce')
    df_msg['CreateDate'] = pd.to_datetime(df_msg['CreateDate'], errors='coerce')
    df_txn['month'] = df_txn['CLOSEDATE'].dt.to_period('M')
    df_msg['month'] = df_msg['CreateDate'].dt.to_period('M')

    txn_stats = df_txn.groupby(['HubSpotId', 'month'], as_index=False).agg(topup_cnt=('TOPUP', 'sum'), topup_flag=('TOPUP', 'max'))
    txn_last = df_txn.sort_values('CLOSEDATE').groupby(['HubSpotId', 'month'], as_index=False).last()
    monthly_txn = txn_stats.merge(txn_last, on=['HubSpotId', 'month'], how='left')

    msg_stats = df_msg.groupby(['AssociatedContactId', 'month'], as_index=False).agg(mail_cnt=('CreateDate', 'count'), last_mail_date=('CreateDate', 'max'))
    msg_last = df_msg.sort_values('CreateDate').groupby(['AssociatedContactId', 'month'], as_index=False).last()
    monthly_msg = msg_stats.merge(msg_last, on=['AssociatedContactId', 'month'], how='left')

    all_clients = pd.unique(pd.concat([df_txn['HubSpotId'], df_msg['AssociatedContactId']]))
    min_month = min(df_txn['month'].min(), df_msg['month'].min())
    max_month = max(df_txn['month'].max(), df_msg['month'].max())
    grid = pd.DataFrame(list(product(all_clients, pd.period_range(min_month, max_month, freq='M'))), columns=['HubSpotId', 'month'])

    panel_df = (grid.merge(monthly_txn, on=['HubSpotId', 'month'], how='left')
                .merge(monthly_msg, left_on=['HubSpotId', 'month'], right_on=['AssociatedContactId', 'month'], how='left', suffixes=('', '_msg'))
                .drop(columns=['AssociatedContactId']).fillna({'topup_cnt': 0, 'topup_flag': 0, 'mail_cnt': 0})
                .sort_values(['HubSpotId', 'month']).reset_index(drop=True))
    panel_df[['topup_cnt', 'topup_flag', 'mail_cnt']] = panel_df[['topup_cnt', 'topup_flag', 'mail_cnt']].astype(int)
    panel_df['target_next_month_TOPUP'] = panel_df.groupby('HubSpotId')['TOPUP'].shift(-1)
    panel_df = panel_df[panel_df['month'] >= '2021-01']
    logging.info(f'Panel created: {len(panel_df)} rows, {panel_df["HubSpotId"].nunique()} clients')

    logging.info('Preparing PM1 risk data...')
    OG_df_Risk_Profile = OG_df_Risk_Profile.merge(df_contact_filtered[['OnboardingId', 'HubSpotId']], how='left', left_on='MandateId', right_on='OnboardingId').drop(columns='OnboardingId')
    df_calls = OG_df_calls[['AssociatedContactId', 'CreateDate', 'CallNotesPreview']]
    df_marketing_email = OG_df_marketing_email
    df_sales_email = OG_df_sales_email[['AssociatedContactId', 'CreateDate', 'EmailBodyPreview']]
    df_PM1 = OG_df_Risk_Profile[['HubSpotId', 'PoA', 'RiskProfile', 'RelationshipManager']]
    logging.info('Features prepared')

    logging.info('Merging call notes...')
    df_calls['CreateDate'] = pd.to_datetime(df_calls['CreateDate'], errors='coerce')
    df_calls['Month'] = df_calls['CreateDate'].dt.to_period('M').astype(str)
    df_calls_pivot = df_calls.groupby(['AssociatedContactId', 'Month'])['CallNotesPreview'].apply(lambda x: '; '.join(x.dropna().astype(str))).reset_index()
    panel_df['month'] = panel_df['month'].astype(str)
    panel_df = panel_df.merge(df_calls_pivot[['AssociatedContactId', 'Month', 'CallNotesPreview']], how='left', left_on=['HubSpotId', 'month'], right_on=['AssociatedContactId', 'Month']).drop(columns=['AssociatedContactId', 'Month'])
    logging.info('Call notes merged')

    logging.info('Merging marketing emails...')
    df_marketing_email['createdAt'] = pd.to_datetime(df_marketing_email['createdAt'], errors='coerce')
    df_marketing_email['Month'] = df_marketing_email['createdAt'].dt.to_period('M').astype(str)
    df_marketing_email_pivot = df_marketing_email.groupby('Month')['emailSubject'].apply(lambda x: '; '.join(x.dropna().astype(str))).reset_index()
    panel_df = panel_df.merge(df_marketing_email_pivot[['Month', 'emailSubject']], how='left', left_on='month', right_on='Month').drop(columns=['Month'])
    logging.info('Marketing emails merged')

    logging.info('Merging sales emails...')
    df_sales_email['CreateDate'] = pd.to_datetime(df_sales_email['CreateDate'], errors='coerce')
    df_sales_email['Month'] = df_sales_email['CreateDate'].dt.to_period('M').astype(str)
    df_sales_email_pivot = df_sales_email.groupby(['AssociatedContactId', 'Month'])['EmailBodyPreview'].apply(lambda x: '; '.join(x.dropna().astype(str))).reset_index()
    panel_df = panel_df.merge(df_sales_email_pivot[['AssociatedContactId', 'Month', 'EmailBodyPreview']], how='left', left_on=['HubSpotId', 'month'], right_on=['AssociatedContactId', 'Month']).drop(columns=['AssociatedContactId', 'Month'])
    logging.info('Sales emails merged')

    logging.info('Merging PM1 risk profiles...')
    df_PM1_deduped = df_PM1.drop_duplicates(subset='HubSpotId', keep='first')
    panel_df = panel_df.merge(df_PM1_deduped[['HubSpotId', 'PoA', 'RiskProfile', 'RelationshipManager']], how='left', on='HubSpotId')
    logging.info('PM1 data merged')

    logging.info('Calculating last contacted feature...')
    filtered_df_meetings = df_engagement[(df_engagement['AssociatedContactId'].isin(df_client_IDs['associatedcontactid'])) & (df_engagement['MeetingOutcome'] == 'COMPLETED')]
    filtered_df_calls = OG_df_calls[(OG_df_calls['AssociatedContactId'].isin(df_client_IDs['associatedcontactid'])) & (OG_df_calls['callDisposition'] == 'Connected')]
    filtered_df_sales_emails = OG_df_sales_email[(OG_df_sales_email['AssociatedContactId'].isin(df_client_IDs['associatedcontactid'])) & (OG_df_sales_email['EmailDirection'] == 'EMAIL')]
    df = pd.concat([filtered_df_calls[['AssociatedContactId', 'CreateDate']], filtered_df_meetings[['AssociatedContactId', 'CreateDate']], filtered_df_sales_emails[['AssociatedContactId', 'CreateDate']]], ignore_index=True)
    df['CreateDate'] = pd.to_datetime(df['CreateDate'])
    df = df[df['AssociatedContactId'].notna()].sort_values(by=['AssociatedContactId', 'CreateDate']).reset_index(drop=True)
    panel_df['month'] = pd.to_datetime(panel_df['month'], format='%Y-%m')
    panel_df['month_end'] = panel_df['month'] + pd.offsets.MonthEnd(0)
    df['month'] = df['CreateDate'].dt.to_period('M').dt.to_timestamp()
    last_contact = df.groupby(['AssociatedContactId', 'month'])['CreateDate'].max().reset_index().rename(columns={'AssociatedContactId': 'HubSpotId', 'CreateDate': 'last_contact_date'})
    panel_df = panel_df.merge(last_contact, how='left', on=['HubSpotId', 'month']).sort_values(['HubSpotId', 'month'])
    panel_df['last_contact_date'] = panel_df.groupby('HubSpotId')['last_contact_date'].ffill()
    panel_df['last_contacted'] = (panel_df['month_end'] - panel_df['last_contact_date']).dt.days
    panel_df.loc[panel_df['TOPUP'] == 1, 'last_contacted'] = 0
    logging.info('Last contacted calculated')

    logging.info('Adding temporal features...')
    panel_df = panel_df.rename(columns={'emailSubject': 'Marketing email', 'EmailBodyPreview': 'Sales Email'})
    panel_df['month'] = pd.to_datetime(panel_df['month'], errors='coerce')
    panel_df['month_num'] = panel_df['month'].dt.month
    panel_df['month_name'] = panel_df['month'].dt.strftime('%B')
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
    panel_df['season'] = panel_df['month_num'].apply(get_season)
    logging.info('Temporal features added')

    logging.info('Selecting final columns...')
    final_columns = ['HubSpotId', 'month', 'TOPUPAMOUNT', 'MeetingNotesPreview', 'ShareOfWalletPotential', 'EngagementLabel', 'HsPersona', 'AgeRange', 'Gender', 'JobTitle', 'TypeOfReferral', 'ShariaCompliant', 'RelationshipManager', 'RiskLevelScore', 'month_num', 'month_name', 'season', 'CallNotesPreview', 'Marketing email', 'Sales Email', 'TOPUP', 'RiskProfile', 'last_contacted', 'ContactOwner']
    existing_columns = [col for col in final_columns if col in panel_df.columns]
    panel_df = panel_df[existing_columns]
    panel_df['created_at'] = datetime.now()
    panel_df['pipeline_run_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.info(f'Final dataset: {len(panel_df)} rows, {len(panel_df.columns)} columns')
    logging.info(f'Unique clients: {panel_df["HubSpotId"].nunique()}')
    logging.info(f'Date range: {panel_df["month"].min()} to {panel_df["month"].max()}')

    save_to_sql_database(panel_df, target_engine, OUTPUT_SCHEMA, OUTPUT_TABLE)
    logging.info(f'Successfully saved {len(panel_df)} rows to {OUTPUT_SCHEMA}.{OUTPUT_TABLE}')
    logging.info('Data Preparation Pipeline completed successfully')
