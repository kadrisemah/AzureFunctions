import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, Table, Column, MetaData, text
from sqlalchemy.types import Integer, BigInteger, Float, String, DateTime, Boolean, NVARCHAR
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import re
import warnings
import os
import unicodedata
import azure.functions as func

# NLTK imports for advanced text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sentence-BERT for semantic embeddings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    logging.warning('NLTK download failed, will retry during execution')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Initialize Sentence-BERT model (lazy loading)
_sbert_model = None

def get_sbert_model():
    """Lazy load Sentence-BERT model"""
    global _sbert_model
    if _sbert_model is None:
        try:
            logging.info('Loading Sentence-BERT model (all-MiniLM-L6-v2)...')
            _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info('Sentence-BERT model loaded successfully')
        except Exception as e:
            logging.warning(f'Failed to load SBERT model: {e}. Falling back to TF-IDF only.')
            _sbert_model = None
    return _sbert_model


def create_db_connection(server, database, username, password):
    """Create database connection using ODBC Driver 18 or 17"""
    driver = 'ODBC Driver 18 for SQL Server'
    connection_string = (
        f'mssql+pyodbc://{username}:{quote_plus(password)}@{server}:1433/{database}'
        f'?driver={driver.replace(" ", "+")}'
    )
    return create_engine(connection_string)


def infer_sqlalchemy_type(dtype, column_name=None, series=None):
    """Infer SQLAlchemy type from pandas dtype"""
    dtype_str = str(dtype)
    if 'datetime' in dtype_str:
        return DateTime
    if dtype_str.startswith('int'):
        # Check if values exceed INT max (2,147,483,647) - use BIGINT if needed
        if series is not None:
            max_val = series.max()
            min_val = series.min()
            if pd.notna(max_val) and pd.notna(min_val):
                if max_val > 2147483647 or min_val < -2147483648:
                    return BigInteger
        return Integer
    if dtype_str.startswith('float'):
        return Float
    if dtype_str == 'bool':
        return Boolean
    if dtype_str == 'object':
        long_text_columns = ['meetingnotespreview', 'callnotespreview', 'notes']
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
    metadata.create_all(engine)
    return table


def save_to_sql_database(df, engine, schema, table_name):
    """Save DataFrame to SQL database"""
    df.columns = [col.replace(' ', '_').replace('-', '_') for col in df.columns]

    # Remove duplicate columns (case-insensitive) - SQL Server is case-insensitive
    original_cols = len(df.columns)
    df = df.loc[:, ~df.columns.str.lower().duplicated()]
    if len(df.columns) < original_cols:
        logging.info(f'[save_to_sql] Removed {original_cols - len(df.columns)} duplicate columns')

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


# Contractions dictionary for text normalization
_contractions = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'ll": " will",
    "'d": " would",
    "'ve": " have",
    "'m": " am"
}

def expand_contractions(text):
    """Expand contractions in text"""
    for c, r in _contractions.items():
        text = re.sub(c, r, text, flags=re.IGNORECASE)
    return text

def clean_text(text, min_len=2):
    """Clean and normalize text for NLP processing with NLTK"""
    if pd.isna(text):
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", str(text))
    # Remove html tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # Expand contractions
    text = expand_contractions(text)
    # Remove non-alpha (keep spaces)
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Tokenise, remove stopwords, lemmatize
    tokens = [t for t in text.split() if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) >= min_len]
    return " ".join(tokens)


def map_utm_campaign_first_touch(x):
    """Standardize UTM campaign names"""
    x = str(x).lower().strip()
    filters = {
        'retire': 'retire', 'diverse': 'diverse', 'direct traffic': 'direct traffic',
        'osol': 'osol', 'club': 'club', 'aos': 'aos', 'organic': 'organic',
        'professional': 'professional', 'argaam': 'argaam', 'lookalike': 'lookalike'
    }
    for key, value in filters.items():
        if key in x:
            return value
    return 'others'


def preprocess_contact_data(df):
    """Preprocess contact data with feature engineering"""
    logging.info('Preprocessing contact data...')
    df = df.copy()

    # Step 1: Impute missing values
    cols_with_zero_fill = [
        'PageDirectoryPortfolioPlanner', 'PageDirectoryRetirementPlanning',
        'PageDirectoryMarketOutlook', 'PageDirectoryArticles',
        'PageDirectoryWhitepaper', 'PageDirectoryDiversification',
        'PageDirectoryInvestorTypeSurvey', 'PageDirectoryInvestmentPlanner', 'AgeRange'
    ]
    for col in cols_with_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    cols_with_default_fill = {
        'WasTheProspectInterestedInInvestment': 'Cannot be determined',
        'WasTheCampaignMessageUnderstood': 'Cannot be determined',
        'MaritalStatus': 'Cannot be determined',
        'PreferredLanguage': 'English',
        'City': 'Cannot be Determined'
    }
    df.fillna(cols_with_default_fill, inplace=True)

    if 'City' in df.columns:
        df['City'] = df['City'].astype(str).str.title().fillna('Cannot be Determined')

    # Step 2: Standardize categorical columns
    if 'UTMCampaignFirstTouch' in df.columns:
        df['UTMCampaignFirstTouch'] = df['UTMCampaignFirstTouch'].apply(map_utm_campaign_first_touch)

    # FirstPageSeen patterns
    if 'FirstPageSeen' in df.columns:
        patterns = {
            r'.*linkedin.*': 'linkedin',
            r'.*client.*': 'Client App',
            r'.*my.tfoco.*': 'PJ',
            r'.*tfoco.*': 'Website',
            r'.*tfo website.*': 'Website',
            r'.*facebook.*': 'Facebook'
        }
        df['FirstPageSeen'] = df['FirstPageSeen'].astype(str)
        for pattern, replacement in patterns.items():
            df['FirstPageSeen'] = df['FirstPageSeen'].replace(pattern, replacement, regex=True)

    # Gender mapping
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({'Yes': 'Male', 'No': 'Female'})

    # Nationality mapping
    if 'Nationality' in df.columns:
        df['Nationality'] = df['Nationality'].astype(str).str.lower().str.strip()
        nationality_mappings = {
            'saudi': 'saudi arabia', 'saudi arabian': 'saudi arabia', 'sa': 'saudi arabia',
            'saudi araboa': 'saudi arabia', 'saudi rabia': 'saudi arabia',
            'sudi': 'saudi arabia', 'saudi_arabian': 'saudi arabia',
            'ksa': 'saudi arabia', 'ٍsaudi': 'saudi arabia', 'ٍsa': 'saudi arabia',
            'sdudi': 'saudi arabia',
            'bahraini': 'bahrain', 'bahrein': 'bahrain'
        }
        df['Nationality'] = df['Nationality'].replace(nationality_mappings)

        # Create binary nationality columns (top 3)
        df['Nationality_bahrain'] = df['Nationality'].str.contains('bahrain', case=False, na=False).astype(int)
        df['Nationality_indian'] = df['Nationality'].str.contains('indian', case=False, na=False).astype(int)
        df['Nationality_saudi arabia'] = df['Nationality'].str.contains('saudi arabia', case=False, na=False).astype(int)

    # Step 3: One-hot encoding
    columns_to_encode = [
        'MaritalStatus', 'Gender', 'CampaignType', 'PreferredLanguage',
        'UTMCampaignFirstTouch', 'UTMSourceFirstTouch',
        'WasTheCampaignMessageUnderstood', 'WasTheProspectInterestedInInvestment',
        'ContactType'
    ]
    available_cols = [col for col in columns_to_encode if col in df.columns]
    if available_cols:
        df_encoded = pd.get_dummies(df[available_cols], prefix=available_cols, dtype=int)
        df = pd.concat([df, df_encoded], axis=1)

    # Encode SegmentClass
    if 'SegmentClass' in df.columns:
        for i in range(5):
            df[f'SegmentClass_{i}'] = (df['SegmentClass'] == i).astype(int)

    # Top 5 encoding for high cardinality features
    if 'Country' in df.columns:
        top_5_countries = df['Country'].value_counts().nlargest(5).index
        df['Country'] = df['Country'].apply(lambda x: x if x in top_5_countries else 'Other')
        df = pd.concat([df, pd.get_dummies(df['Country'], prefix='Country')], axis=1)

    if 'FirstPageSeen' in df.columns:
        top_5_first_pages = df['FirstPageSeen'].value_counts().nlargest(5).index
        df['FirstPageSeen'] = df['FirstPageSeen'].apply(lambda x: x if x in top_5_first_pages else 'Other')
        df = pd.concat([df, pd.get_dummies(df['FirstPageSeen'], prefix='FirstPageSeen')], axis=1)

    # Engagement label encoding
    if 'EngagementLabel' in df.columns:
        engagement_label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df['engagement_label'] = df['EngagementLabel'].map(engagement_label_mapping).fillna(-1).astype(int)

    # Age range encoding
    if 'AgeRange' in df.columns:
        df['age_range_below_21'] = (df['AgeRange'] == 'Below 20').astype(int)
        df['age_range_21_30'] = (df['AgeRange'] == '21 - 30').astype(int)
        df['age_range_31_40'] = (df['AgeRange'] == '31 - 40').astype(int)
        df['age_range_41_50'] = (df['AgeRange'] == '41 - 50').astype(int)
        df['age_range_51_60'] = (df['AgeRange'] == '51 - 60').astype(int)
        df['age_range_over_60'] = (df['AgeRange'] == 'Above 60').astype(int)

    # CreateDate bins
    if 'CreateDate' in df.columns:
        df['CreateDate'] = pd.to_datetime(df['CreateDate'], errors='coerce')
        start_date = df['CreateDate'].min()
        if pd.notna(start_date):
            bins = [
                start_date,
                start_date + timedelta(days=547),
                start_date + timedelta(days=730),
                start_date + timedelta(days=912),
                start_date + timedelta(days=1095),
                datetime.today()
            ]
            labels = ['0 to 1.5 years ago', '1.5 to 2 years ago', '2 to 2.5 years ago',
                     '2.5 to 3 years ago', '3+ years ago']
            df['CreateDate_Bins'] = pd.cut(df['CreateDate'], bins=bins, labels=labels, right=False)
            df = pd.concat([df, pd.get_dummies(df['CreateDate_Bins'], prefix='Created')], axis=1)

    # Target variable
    if 'LeadStageNewCEO' in df.columns:
        df['Target'] = df['LeadStageNewCEO'].str.contains('SAL', case=False, na=False).astype(int)

    logging.info(f'Preprocessing complete: {len(df)} rows, {len(df.columns)} columns')
    return df


def process_meetings_data(meetings_df, valid_hubspot_ids):
    """Process meetings engagement data with NLP clustering"""
    logging.info('Processing meetings data...')

    if meetings_df is None or len(meetings_df) == 0:
        logging.warning('No meetings data available')
        return pd.DataFrame()

    meetings_df = meetings_df.copy()
    meetings_df.rename(columns={'AssociatedContactId': 'HubSpotId'}, inplace=True)
    meetings_df = meetings_df[meetings_df['HubSpotId'].isin(valid_hubspot_ids)]

    # Date normalization
    if 'MeetingOutcome' in meetings_df.columns and 'CreateDate' in meetings_df.columns:
        meetings_df.loc[meetings_df['MeetingOutcome'] == 'SCHEDULED', 'ActivityDate'] = \
            meetings_df.loc[meetings_df['MeetingOutcome'] == 'SCHEDULED', 'CreateDate']

    if 'ActivityDate' in meetings_df.columns:
        meetings_df['ActivityDate'] = pd.to_datetime(meetings_df['ActivityDate']).dt.date

    # Text cleaning
    if 'MeetingNotesPreview' in meetings_df.columns:
        meetings_df['MeetingNotes_clean'] = meetings_df['MeetingNotesPreview'].apply(clean_text)
        meetings_df['HasNotes'] = meetings_df['MeetingNotes_clean'].str.strip().astype(bool)

        # TF-IDF vectorization (as fallback)
        notes_with_content = meetings_df[meetings_df['HasNotes']]
        if len(notes_with_content) > 10:
            try:
                # Try Sentence-BERT embeddings first (better semantic representation)
                sbert_model = get_sbert_model()
                if sbert_model is not None:
                    logging.info('Generating Sentence-BERT embeddings for meeting notes...')
                    embeddings = sbert_model.encode(
                        notes_with_content['MeetingNotes_clean'].tolist(),
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    logging.info(f'SBERT embeddings generated: shape {embeddings.shape}')
                else:
                    # Fallback to TF-IDF if SBERT unavailable
                    logging.info('Using TF-IDF embeddings (SBERT not available)')
                    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.9, ngram_range=(1,2))
                    tfidf_matrix = tfidf_vectorizer.fit_transform(notes_with_content['MeetingNotes_clean'])
                    embeddings = normalize(tfidf_matrix).toarray()

                # Clustering
                n_clusters = 3
                cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
                cluster_labels = cluster_model.fit_predict(embeddings)
                meetings_df.loc[notes_with_content.index, 'NoteSegment'] = cluster_labels

                # Segment counts per contact
                seg_count = meetings_df.groupby(['HubSpotId', 'NoteSegment']).size().unstack(fill_value=0)
                seg_count.columns = [f"SegmentCount_{int(c)}" for c in seg_count.columns]
                seg_count = seg_count.reset_index()
            except Exception as e:
                logging.warning(f'Clustering failed: {e}')
                seg_count = None
        else:
            seg_count = None
    else:
        seg_count = None

    # Aggregation by HubSpotId
    agg_dict = {}
    if 'MeetingOutcome' in meetings_df.columns:
        agg_dict['ScheduledMeetings'] = ('MeetingOutcome', lambda x: (x == 'SCHEDULED').sum())
        agg_dict['CompletedMeetings'] = ('MeetingOutcome', lambda x: (x == 'COMPLETED').sum())

    if agg_dict:
        pivot_meetings_df = meetings_df.groupby('HubSpotId').agg(**agg_dict).reset_index()
    else:
        pivot_meetings_df = meetings_df[['HubSpotId']].drop_duplicates()
        pivot_meetings_df['ScheduledMeetings'] = 0
        pivot_meetings_df['CompletedMeetings'] = 0

    # Merge segment counts if available
    if seg_count is not None:
        pivot_meetings_df = pivot_meetings_df.merge(seg_count, on='HubSpotId', how='left')
        for i in range(3):
            col = f'SegmentCount_{i}'
            if col not in pivot_meetings_df.columns:
                pivot_meetings_df[col] = 0
    else:
        for i in range(3):
            pivot_meetings_df[f'SegmentCount_{i}'] = 0

    # Get top 5 most recent meeting notes per contact (representative notes)
    if 'MeetingNotesPreview' in meetings_df.columns and 'ActivityDate' in meetings_df.columns:
        meetings_df_sorted = meetings_df.sort_values(by=['HubSpotId', 'ActivityDate'], ascending=[True, False])

        def clean_text_for_llm(text):
            """Clean text for LLM input: remove line breaks, special chars, extra spaces"""
            if pd.isna(text):
                return ""
            text = re.sub(r'\s+', ' ', text)  # replace multiple whitespace/newlines
            text = re.sub(r'[^\w\s.,!?-]', '', text)  # remove special characters
            return text.strip()

        def get_top_notes(group):
            notes = group['MeetingNotesPreview'].dropna().head(5).apply(clean_text_for_llm)
            return '\n\n'.join(notes)

        top_notes_df = (
            meetings_df_sorted
            .groupby('HubSpotId', as_index=False)
            .apply(get_top_notes)
        )
        top_notes_df.columns = ['HubSpotId', 'TopSegmentRepresentative']

        # Merge to pivot_meetings_df
        pivot_meetings_df = pivot_meetings_df.merge(top_notes_df, on='HubSpotId', how='left')

    logging.info(f'Meetings processed: {len(pivot_meetings_df)} unique contacts')
    return pivot_meetings_df


def process_calls_data(calls_df, valid_hubspot_ids):
    """Process calls engagement data with NLP clustering"""
    logging.info('Processing calls data...')

    if calls_df is None or len(calls_df) == 0:
        logging.warning('No calls data available')
        return pd.DataFrame()

    calls_df = calls_df.copy()
    calls_df.rename(columns={'AssociatedContactId': 'HubSpotId'}, inplace=True)
    calls_df = calls_df[calls_df['HubSpotId'].isin(valid_hubspot_ids)]

    if 'ActivityDate' in calls_df.columns:
        calls_df['ActivityDate'] = pd.to_datetime(calls_df['ActivityDate']).dt.date

    # Text cleaning and clustering
    if 'CallNotesPreview' in calls_df.columns:
        calls_df['CallNotes_clean'] = calls_df['CallNotesPreview'].apply(clean_text)
        calls_df['HasCallNotes'] = calls_df['CallNotes_clean'].str.strip().astype(bool)

        # Clustering for call notes (same as meetings)
        calls_with_notes = calls_df[calls_df['HasCallNotes']]
        if len(calls_with_notes) > 10:
            try:
                # Try Sentence-BERT embeddings first
                sbert_model = get_sbert_model()
                if sbert_model is not None:
                    logging.info('Generating Sentence-BERT embeddings for call notes...')
                    embeddings_calls = sbert_model.encode(
                        calls_with_notes['CallNotes_clean'].tolist(),
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    logging.info(f'SBERT call embeddings generated: shape {embeddings_calls.shape}')
                else:
                    # Fallback to TF-IDF
                    logging.info('Using TF-IDF embeddings for calls (SBERT not available)')
                    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.9, ngram_range=(1,2))
                    tfidf_matrix = tfidf_vectorizer.fit_transform(calls_with_notes['CallNotes_clean'])
                    embeddings_calls = normalize(tfidf_matrix).toarray()

                # Clustering
                n_clusters = 3
                cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
                cluster_labels = cluster_model.fit_predict(embeddings_calls)
                calls_df.loc[calls_with_notes.index, 'CallNoteSegment'] = cluster_labels
            except Exception as e:
                logging.warning(f'Call clustering failed: {e}')
                calls_df['CallNoteSegment'] = -1
        else:
            calls_df['CallNoteSegment'] = -1
    else:
        calls_df['CallNoteSegment'] = -1

    # Aggregation
    agg_dict = {}
    if 'callDisposition' in calls_df.columns:
        agg_dict['CallAttempts'] = ('callDisposition', 'count')
        agg_dict['ConnectedCalls'] = ('callDisposition', lambda x: (x == 'Connected').sum())

    if agg_dict:
        pivot_calls_df = calls_df.groupby('HubSpotId').agg(**agg_dict).reset_index()
    else:
        pivot_calls_df = calls_df[['HubSpotId']].drop_duplicates()
        pivot_calls_df['CallAttempts'] = 0
        pivot_calls_df['ConnectedCalls'] = 0

    # Get top 5 most recent call notes per contact (representative notes)
    if 'CallNotes_clean' in calls_df.columns and 'ActivityDate' in calls_df.columns:
        calls_df_sorted = calls_df.sort_values(by=['HubSpotId', 'ActivityDate'], ascending=[True, False])

        def clean_text_for_llm(text):
            """Clean text for LLM input: remove line breaks, special chars, extra spaces"""
            if pd.isna(text):
                return ""
            text = re.sub(r'\s+', ' ', text)  # replace multiple whitespace/newlines
            text = re.sub(r'[^\w\s.,!?-]', '', text)  # remove special characters
            return text.strip()

        def get_top_notes(group):
            notes = group['CallNotes_clean'].dropna().head(5).apply(clean_text_for_llm)
            return '\n\n'.join(notes)

        top_notes_df = (
            calls_df_sorted
            .groupby('HubSpotId', as_index=False)
            .apply(get_top_notes)
        )
        top_notes_df.columns = ['HubSpotId', 'TopSegmentRepresentativeCalls']

        # Merge to pivot_calls_df
        pivot_calls_df = pivot_calls_df.merge(top_notes_df, on='HubSpotId', how='left')

    logging.info(f'Calls processed: {len(pivot_calls_df)} unique contacts')
    return pivot_calls_df


def filter_high_missing_data(df):
    """Filter out contacts with 80%+ missing categorical features"""
    logging.info('Filtering contacts with high missing data...')

    columns_grouped = {
        'one_hot_encoded': {
            'age_range': ['age_range_below_21', 'age_range_21_30', 'age_range_31_40',
                          'age_range_41_50', 'age_range_51_60', 'age_range_over_60'],
            'MaritalStatus': ['MaritalStatus_Divorced', 'MaritalStatus_Institution',
                              'MaritalStatus_Married', 'MaritalStatus_Single'],
            'Gender': ['Gender_Female', 'Gender_Institution', 'Gender_Male'],
            'CampaignType': ['CampaignType_Awareness', 'CampaignType_Lead Gen'],
            'PreferredLanguage': ['PreferredLanguage_Arabic', 'PreferredLanguage_English'],
            'UTMCampaignFirstTouch': ['UTMCampaignFirstTouch_aos', 'UTMCampaignFirstTouch_argaam',
                                      'UTMCampaignFirstTouch_club', 'UTMCampaignFirstTouch_direct traffic',
                                      'UTMCampaignFirstTouch_diverse', 'UTMCampaignFirstTouch_lookalike',
                                      'UTMCampaignFirstTouch_organic', 'UTMCampaignFirstTouch_osol',
                                      'UTMCampaignFirstTouch_others', 'UTMCampaignFirstTouch_professional',
                                      'UTMCampaignFirstTouch_retire'],
            'UTMSourceFirstTouch': ['UTMSourceFirstTouch_Argaam', 'UTMSourceFirstTouch_Client App',
                                    'UTMSourceFirstTouch_Direct traffic', 'UTMSourceFirstTouch_Email Marketing',
                                    'UTMSourceFirstTouch_Facebook', 'UTMSourceFirstTouch_GDN',
                                    'UTMSourceFirstTouch_GMB', 'UTMSourceFirstTouch_Instagram',
                                    'UTMSourceFirstTouch_Integration', 'UTMSourceFirstTouch_Linkedin',
                                    'UTMSourceFirstTouch_Offline (import)', 'UTMSourceFirstTouch_Offline (sales contacts)',
                                    'UTMSourceFirstTouch_Old Webinar (Integration)', 'UTMSourceFirstTouch_Organic',
                                    'UTMSourceFirstTouch_Referrals', 'UTMSourceFirstTouch_Scheduler App',
                                    'UTMSourceFirstTouch_Search', 'UTMSourceFirstTouch_Snapchat',
                                    'UTMSourceFirstTouch_Social media (organic)', 'UTMSourceFirstTouch_Twitter',
                                    'UTMSourceFirstTouch_Whatsapp', 'UTMSourceFirstTouch_tfo-website'],
            'Country': ['Country_Bahrain', 'Country_Kuwait', 'Country_Other',
                        'Country_Saudi Arabia', 'Country_United Arab Emirates'],
            'FirstPageSeen': ['FirstPageSeen_Facebook', 'FirstPageSeen_Other',
                              'FirstPageSeen_PJ', 'FirstPageSeen_Website', 'FirstPageSeen_linkedin'],
            'Nationality': ['Nationality_bahrain', 'Nationality_indian', 'Nationality_saudi arabia'],
        }
    }

    # Check for missing values (one-hot encoded groups)
    missing_values_df = pd.DataFrame(index=df.index)

    # Missing in one-hot groups (all columns in a group are 0 = missing)
    for group_name, group_columns in columns_grouped['one_hot_encoded'].items():
        # Filter to only existing columns
        existing_cols = [col for col in group_columns if col in df.columns]
        if existing_cols:
            missing_values_df[group_name] = (df[existing_cols].sum(axis=1) == 0)
        else:
            # If no columns exist, assume not missing
            missing_values_df[group_name] = False

    # Calculate the percentage of missing values for each row
    missing_percentage = missing_values_df.mean(axis=1)

    # Filter rows with 80% or more missing values
    high_missing_condition = missing_percentage >= 0.8

    # Create a new DataFrame with the filtered rows
    high_missing_data = df[high_missing_condition]

    # Creating high_missing_ids list
    if 'HubSpotId' in df.columns:
        high_missing_ids = high_missing_data['HubSpotId'].to_list()
    else:
        high_missing_ids = high_missing_data.index.to_list()

    logging.info(f'Found {len(high_missing_ids)} contacts with 80%+ missing categorical features')

    return high_missing_data, high_missing_ids


def main(mytimer: func.TimerRequest) -> None:
    logging.info('SQLSAL Data Preparation Function started')

    OUTPUT_SCHEMA = 'dbo'
    OUTPUT_TABLE = 'sqlsal_training_dataset'

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
        logging.error('Missing required environment variables')
        raise ValueError('Missing required environment variables')

    logging.info('Credentials retrieved successfully')

    logging.info('Establishing database connections...')
    source_engine = create_db_connection(source_server, source_database, source_username, source_password)
    target_engine = create_db_connection(target_server, target_database, target_username, target_password)
    logging.info('Database connections established')

    # Extract SQL/SAL contact data
    logging.info('Extracting SQL/SAL contact data...')
    sql_sal_query = """
    SELECT *
    FROM vi_HSContact
    WHERE LeadStageNewCEO IN ('SQL 1', 'SQL 2', 'SAL 1', 'SAL 2')
    AND IsThisLeadAlreadyFoundInAnInternalDatabase = 'Not Found'
    AND ContactType IN ('DFO', 'TFO')
    """
    sal_df = pd.read_sql(sql_sal_query, source_engine)
    logging.info(f'SQL/SAL contacts extracted: {len(sal_df)} records')

    # Extract meetings data
    logging.info('Extracting meetings engagement data...')
    meetings_query = """
    SELECT AssociatedContactId, CreateDate, ActivityDate, MeetingOutcome, MeetingNotesPreview
    FROM vi_HSEngagementmeeting
    """
    meetings_df = pd.read_sql(meetings_query, source_engine)
    logging.info(f'Meetings extracted: {len(meetings_df)} records')

    # Extract calls data
    logging.info('Extracting calls engagement data...')
    calls_query = """
    SELECT AssociatedContactId, CreateDate, ActivityDate, callDisposition, CallNotesPreview
    FROM vi_HSEngagementCall
    """
    calls_df = pd.read_sql(calls_query, source_engine)
    logging.info(f'Calls extracted: {len(calls_df)} records')

    # Preprocess contact data
    processed_df = preprocess_contact_data(sal_df)

    # Drop columns with excessive missing data or non-useful for modeling
    columns_to_drop = [
        "PersonId", "ClientType", "BirthPlace", "GenderIdentity", "BirthCountry",
        "BecameAnOpportunityDate", "ClickedAnyOtherLinkLeadGenEmail",
        "ClickedPjLinkLeadGenEmail", "DealInterestedProspectJourney",
        "DoYouHaveSignificantExposureToAnyAssetsYouWouldLikeExcludedFromYourPortfolio",
        "IsThisLeadAlreadyFoundInAnInternalDatabase", "LastTransitionDate",
        "OpenedLeadGenEmail", "ReceivedLeadGenEmail",
        "SignUpDateLeadGenAndProspectJourney", "SimulatorFlow",
        "WhatAreYourGoalsForThisInvestmentOtherDetails", "AreYouAnAccreditedInvestor",
        "PayInInstallments", "NumberofPageViews", "OriginalSource", "LastPageSeen",
        "ReasonForDisqualification", "RecentSalesEmailOpenedDate",
        "RecentSalesEmailClickedDate", "OtherCommentPA", "OtherCommentsRM",
        "AreYouAQualifiedInvestorExecutionFlow", "LastContacted",
        "DoesTheEmaiLookReal", "CouldTheLeadBeaRelevantClientForDFOOrTFO",
        "ClearOutPhoneCountryName", "RetailClient"
    ]
    existing_drop_cols = [col for col in columns_to_drop if col in processed_df.columns]
    if existing_drop_cols:
        processed_df = processed_df.drop(columns=existing_drop_cols)
        logging.info(f'Dropped {len(existing_drop_cols)} unnecessary columns')

    # Final column drops (before creating final dataset)
    final_drops = ["DAYSTOCLOSE", "ValidFrom", "ValidUntil", "ContactOwnerId"]
    existing_final_drops = [col for col in final_drops if col in processed_df.columns]
    if existing_final_drops:
        processed_df = processed_df.drop(columns=existing_final_drops)
        logging.info(f'Dropped {len(existing_final_drops)} final columns')

    # Filter out 'Lead' stage
    if 'LeadStageNewCEO' in processed_df.columns:
        original_len = len(processed_df)
        processed_df = processed_df[~(processed_df['LeadStageNewCEO'] == 'Lead')]
        logging.info(f'Filtered out Leads: {original_len - len(processed_df)} records removed')

    # Get valid HubSpot IDs
    valid_hubspot_ids = processed_df['HubSpotId'].unique().tolist()

    # Process meetings and calls
    pivot_meetings_df = process_meetings_data(meetings_df, valid_hubspot_ids)
    pivot_calls_df = process_calls_data(calls_df, valid_hubspot_ids)

    # Merge engagement features
    if not pivot_meetings_df.empty:
        processed_df = processed_df.merge(pivot_meetings_df, on='HubSpotId', how='left')
        meetings_cols = ['ScheduledMeetings', 'CompletedMeetings', 'SegmentCount_0', 'SegmentCount_1', 'SegmentCount_2']
        for col in meetings_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(0).round().astype(int)

    if not pivot_calls_df.empty:
        processed_df = processed_df.merge(pivot_calls_df, on='HubSpotId', how='left')
        calls_cols = ['CallAttempts', 'ConnectedCalls']
        for col in calls_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(0).round().astype(int)

    # Filter out contacts with high missing data (80%+ missing categorical features)
    high_missing_data, high_missing_ids = filter_high_missing_data(processed_df)
    if high_missing_ids:
        processed_df = processed_df[~processed_df['HubSpotId'].isin(high_missing_ids)]
        logging.info(f'Removed {len(high_missing_ids)} contacts with high missing data')

    # Add metadata
    processed_df['created_at'] = datetime.now()
    processed_df['pipeline_run_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.info(f'Final dataset: {len(processed_df)} rows, {len(processed_df.columns)} columns')
    logging.info(f'Unique contacts: {processed_df["HubSpotId"].nunique()}')
    if 'Target' in processed_df.columns:
        logging.info(f'Target distribution: {processed_df["Target"].value_counts().to_dict()}')

    # Remove duplicate columns (case-insensitive) - SQL Server column names are case-insensitive
    original_cols = len(processed_df.columns)
    processed_df = processed_df.loc[:, ~processed_df.columns.str.lower().duplicated()]
    if len(processed_df.columns) < original_cols:
        logging.info(f'Removed {original_cols - len(processed_df.columns)} duplicate columns (case-insensitive)')

    # Save to SQL database
    save_to_sql_database(processed_df, target_engine, OUTPUT_SCHEMA, OUTPUT_TABLE)
    logging.info(f'Successfully saved {len(processed_df)} rows to {OUTPUT_SCHEMA}.{OUTPUT_TABLE}')
    logging.info('SQLSAL Data Preparation Pipeline completed successfully')
