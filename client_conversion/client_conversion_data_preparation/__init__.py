import logging
import pandas as pd
import numpy as np
import re
import unicodedata
from datetime import datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import pyodbc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import os
import azure.functions as func
# Sentence-BERT for semantic embeddings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

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


# Text Cleaning Functions for NLP
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

_contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'ll": " will", "'d": " would", "'ve": " have", "'m": " am"
}


def expand_contractions(text):
    for c, r in _contractions.items():
        text = re.sub(c, r, text, flags=re.IGNORECASE)
    return text


def clean_text(text, min_len=2):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = expand_contractions(text)
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    tokens = [t for t in text.split() if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if len(t) >= min_len]
    return " ".join(tokens)


def map_utm_campaign_first_touch(x):
    """Map UTM campaign to categories"""
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


def main(mytimer: func.TimerRequest) -> None:
    logging.info('Client Conversion Data Preparation Function started')

    # Configuration
    INPUT_SCHEMA = 'dbo'
    OUTPUT_TABLE = 'client_conversion_training_dataset'

    # Download NLTK data if not present
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

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

    # Create database connections
    logging.info('Establishing database connections...')
    source_engine = create_db_connection(source_server, source_database, source_username, source_password)
    target_engine = create_db_connection(target_server, target_database, target_username, target_password)
    logging.info('Database connections established')

    # SQL Queries
    client_sal_query = '''
    SELECT *
        FROM vi_HSContact
        WHERE
            (
                OnboardingId IS NOT NULL
                AND LeadStageNew = 'Committed Capital'
                AND OnboardingId NOT IN (5656, 90909, 100000, 111122223333)
                AND (ClientType = 'DFO' OR HubSpotId IN (79683601, 96544351, 142538951, 4348122, 243665151, 248616251))
            )
            OR LeadStageNewCEO IN ('SAL 1', 'SAL 2')
            AND ContactType IN ('DFO', 'TFO');
        '''

    meetings_query = '''
    SELECT AssociatedContactId, CreateDate,
           ActivityDate, MeetingOutcome, MeetingNotesPreview
    FROM vi_HSEngagementmeeting;
    '''

    calls_query = '''
    SELECT AssociatedContactId, CreateDate,
            ActivityDate, callDisposition, CallNotesPreview
            FROM vi_HSEngagementCall;
            '''

    # Load data
    logging.info('Loading client/SAL data from source database...')
    client_df = pd.read_sql(client_sal_query, source_engine)
    logging.info(f'Loaded {len(client_df)} client/SAL records')

    logging.info('Loading meetings data...')
    meetings_df = pd.read_sql(meetings_query, source_engine)
    logging.info(f'Loaded {len(meetings_df)} meeting records')

    logging.info('Loading calls data...')
    calls_df = pd.read_sql(calls_query, source_engine)
    logging.info(f'Loaded {len(calls_df)} call records')

    # Process client_df
    logging.info('Processing client data...')
    client_df.reset_index(inplace=True, drop=True)

    # Convert large int64 columns to string to prevent overflow
    int_cols = client_df.select_dtypes(include=["int64"]).columns.tolist()
    for col in int_cols:
        if client_df[col].max() > 1e14 or client_df[col].min() < -1e14:
            client_df[col] = client_df[col].astype(str)
            logging.debug(f"Converted {col} to string (too large for timestamp)")

    # Drop irrelevant columns
    processed_df = client_df.drop(
        columns=[
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
        ], errors='ignore'
    )

    # Fill missing values
    cols_with_zero_fill = [
        'PageDirectoryPortfolioPlanner', 'PageDirectoryRetirementPlanning',
        'PageDirectoryMarketOutlook', 'PageDirectoryArticles',
        'PageDirectoryWhitepaper', 'PageDirectoryDiversification', 'PageDirectoryInvestorTypeSurvey',
        'PageDirectoryInvestmentPlanner', 'AgeRange'
    ]
    processed_df[cols_with_zero_fill] = processed_df[cols_with_zero_fill].fillna(0)

    cols_with_default_fill = {
        'WasTheProspectInterestedInInvestment': 'Cannot be determined',
        'WasTheCampaignMessageUnderstood': 'Cannot be determined',
        'MaritalStatus': 'Cannot be determined',
        'PreferredLanguage': 'English',
        'City': 'Cannot be Determined'
    }
    processed_df.fillna(cols_with_default_fill, inplace=True)
    processed_df['City'] = processed_df['City'].astype(str).str.title().fillna('Cannot be Determined')

    # UTM Campaign mapping
    processed_df['UTMCampaignFirstTouch'] = processed_df['UTMCampaignFirstTouch'].apply(map_utm_campaign_first_touch)

    # FirstPageSeen mapping
    patterns = {
        r'.*linkedin.*': 'linkedin',
        r'.*client.*': 'Client App',
        r'.*my.tfoco.*': 'PJ',
        r'.*tfoco.*': 'Website',
        r'.*tfo website.*': 'Website',
        r'.*facebook.*': 'Facebook'
    }
    processed_df['FirstPageSeen'] = processed_df['FirstPageSeen'].astype(str)
    for pattern, replacement in patterns.items():
        processed_df['FirstPageSeen'] = processed_df['FirstPageSeen'].replace(pattern, replacement, regex=True)

    # Gender mapping
    processed_df['Gender'] = processed_df['Gender'].replace({'Yes': 'Male', 'No': 'Female'})

    # Nationality mapping
    processed_df['Nationality'] = processed_df['Nationality'].astype(str).str.lower().str.strip()
    nationality_mappings = {
        'saudi': 'saudi arabia', 'saudi arabian': 'saudi arabia', 'sa': 'saudi arabia',
        'saudi araboa': 'saudi arabia', 'saudi rabia': 'saudi arabia',
        'sudi': 'saudi arabia', 'saudi_arabian': 'saudi arabia',
        'ksa': 'saudi arabia', 'ٍsaudi': 'saudi arabia', 'ٍsa': 'saudi arabia',
        'sdudi': 'saudi arabia', 'bahraini': 'bahrain', 'bahrein': 'bahrain',
    }
    processed_df['Nationality'] = processed_df['Nationality'].replace(nationality_mappings)

    # Filter to Saudi Arabia only
    processed_df = processed_df[processed_df['Nationality'] == 'saudi arabia']
    processed_df.reset_index(inplace=True, drop=True)

    # FirstPageSeen top 5 categorization
    top_5_first_pages = processed_df['FirstPageSeen'].value_counts().nlargest(5).index
    processed_df['FirstPageSeen'] = processed_df['FirstPageSeen'].apply(lambda x: x if x in top_5_first_pages else 'Other')

    # Remove 'Lead' stage
    processed_df = processed_df[~(processed_df['LeadStageNewCEO'] == 'Lead')]

    # LeadStageNew mapping
    lead_stage_mapping = {
        'INACTIVE': 0, 'Dormant Lead': 1, 'LOST': 0,
        'Booked Meeting': 3, 'Meeting face 2 face': 6, 'COMMITTED CAPITAL': 12,
        'RISK QUESTIONNAIRE ANSWERED': 4, '60+ DAYS TO CLOSE': 8, '15 DAYS TO CLOSE': 9,
        'OFFER SENT': 10, 'KYC Submitted': 11, 'Account Opened': 11, 'Account Closed': 0,
        'Others': 0
    }
    processed_df['LeadStageNew'] = processed_df['LeadStageNew'].map(lead_stage_mapping)

    logging.info(f'Processed client data: {len(processed_df)} records')

    # Process meetings data with NLP
    logging.info('Processing meetings data with NLP...')
    meetings_df.rename(columns={'AssociatedContactId': 'HubSpotId'}, inplace=True)
    meetings_df = meetings_df[meetings_df['HubSpotId'].isin(processed_df['HubSpotId'])]

    meetings_df.loc[
        meetings_df['MeetingOutcome'] == 'SCHEDULED', 'ActivityDate'
    ] = meetings_df.loc[
        meetings_df['MeetingOutcome'] == 'SCHEDULED', 'CreateDate'
    ]

    meetings_df['ActivityDate'] = pd.to_datetime(meetings_df['ActivityDate']).dt.date
    meetings_df['CreateDate'] = pd.to_datetime(meetings_df['CreateDate'])
    meetings_df['ActivityDate'] = pd.to_datetime(meetings_df['ActivityDate'])

    # Clean meeting notes
    meetings_df['MeetingNotes_clean'] = meetings_df['MeetingNotesPreview'].astype(str).apply(clean_text)
    meetings_df['HasNotes'] = meetings_df['MeetingNotes_clean'].str.strip().astype(bool)

    # Try Sentence-BERT embeddings first (better semantic representation)
    sbert_model = get_sbert_model()
    if sbert_model is not None:
        logging.info('Generating Sentence-BERT embeddings for meeting notes...')
        embeddings = sbert_model.encode(
            meetings_df['MeetingNotes_clean'].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True
        )
        logging.info(f'SBERT embeddings generated: shape {embeddings.shape}')
    else:
        # Fallback to TF-IDF if SBERT unavailable
        logging.info('Using TF-IDF embeddings (SBERT not available)')
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.9, ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(meetings_df['MeetingNotes_clean'])
        embeddings = normalize(tfidf_matrix).toarray()

    # Add meeting features
    outcome_descriptions = {
        'COMPLETED': 'Meeting was successfully completed',
        'SCHEDULED': 'Meeting is scheduled for future',
        'CANCELED': 'Meeting was canceled',
        'RESCHEDULED': 'Meeting was rescheduled',
        'NO_SHOW': 'Contact did not attend',
        None: 'No outcome recorded'
    }
    meetings_df['MeetingOutcomeDescription'] = meetings_df['MeetingOutcome'].map(outcome_descriptions)

    meetings_df['DaysFromCreateToActivity'] = (meetings_df['ActivityDate'] - meetings_df['CreateDate']).dt.days
    meetings_df['DaysSinceActivity'] = (pd.Timestamp.now() - meetings_df['ActivityDate']).dt.days
    meetings_df['DaysSinceCreate'] = (pd.Timestamp.now() - meetings_df['CreateDate']).dt.days

    # Clustering meetings
    logging.info('Clustering meeting notes...')
    n_clusters = 3
    meetings_df['NoteSegment'] = -1

    valid_notes_mask = meetings_df['MeetingNotesPreview'].notna() & (meetings_df['MeetingNotesPreview'].str.strip() != '')
    valid_indices = meetings_df[valid_notes_mask].index.tolist()

    cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=3584)
    cluster_labels = cluster_model.fit_predict(embeddings)

    num_to_assign = min(len(cluster_labels), len(valid_indices))
    indices_to_use = valid_indices[:num_to_assign]
    labels_to_use = cluster_labels[:num_to_assign]

    for idx, label in zip(indices_to_use, labels_to_use):
        meetings_df.at[idx, 'NoteSegment'] = int(label)

    logging.info(f'Assigned {num_to_assign} meeting cluster labels')

    # Aggregate meetings by contact
    seg_count = meetings_df.groupby(['HubSpotId', 'NoteSegment']).size().unstack(fill_value=0)
    seg_count.columns = [f"SegmentCount_{int(c)}" for c in seg_count.columns]
    seg_count = seg_count.reset_index()

    pivot_meetings_df = meetings_df.groupby('HubSpotId').agg(
        ScheduledMeetings=('MeetingOutcome', lambda x: (x == 'SCHEDULED').sum()),
        CompletedMeetings=('MeetingOutcome', lambda x: (x == 'COMPLETED').sum()),
        CanceledMeetings=('MeetingOutcome', lambda x: (x == 'CANCELED').sum()),
        NoShowMeetings=('MeetingOutcome', lambda x: (x == 'NO_SHOW').sum()),
        TotalMeetings=('MeetingOutcome', 'count'),
        AvgDaysFromCreateToActivity=('DaysFromCreateToActivity', 'mean'),
        MedianDaysFromCreateToActivity=('DaysFromCreateToActivity', 'median'),
        DaysSinceLastActivity=('DaysSinceActivity', 'min'),
        DaysSinceFirstActivity=('DaysSinceActivity', 'max'),
        FirstMeetingDate=('ActivityDate', 'min'),
        LastMeetingDate=('ActivityDate', 'max'),
        MeetingSpanDays=('ActivityDate', lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
    ).reset_index()

    pivot_meetings_df['MeetingsPerMonth'] = pivot_meetings_df.apply(
        lambda row: (row['TotalMeetings'] / (row['MeetingSpanDays'] / 30.44))
        if row['MeetingSpanDays'] > 0 else row['TotalMeetings'],
        axis=1
    )
    pivot_meetings_df['CompletionRate'] = (
        pivot_meetings_df['CompletedMeetings'] / pivot_meetings_df['TotalMeetings']
    ).fillna(0)

    pivot_meetings_df = pivot_meetings_df.merge(seg_count, on='HubSpotId', how='left').fillna(0)

    # Add advanced segment features
    logging.info('Creating advanced meeting segment features...')

    # TopSegmentRepresentative and TopSegmentLabel
    def get_top_segment(group):
        """Get the most common segment for each contact"""
        if 'NoteSegment' in group.columns:
            segment_counts = group['NoteSegment'].value_counts()
            if len(segment_counts) > 0:
                top_segment = segment_counts.index[0]
                return pd.Series({
                    'TopSegmentRepresentative': int(top_segment),
                    'TopSegmentLabel': f'Segment_{int(top_segment)}'
                })
        return pd.Series({'TopSegmentRepresentative': -1, 'TopSegmentLabel': 'No_Segment'})

    top_segments = meetings_df.groupby('HubSpotId').apply(get_top_segment).reset_index()
    pivot_meetings_df = pivot_meetings_df.merge(top_segments, on='HubSpotId', how='left')
    pivot_meetings_df['TopSegmentRepresentative'] = pivot_meetings_df['TopSegmentRepresentative'].fillna(-1).astype(int)
    pivot_meetings_df['TopSegmentLabel'] = pivot_meetings_df['TopSegmentLabel'].fillna('No_Segment')

    # Sample notes from each segment (first note from each segment per contact)
    for seg_id in [-1, 0, 1, 2]:
        segment_notes = meetings_df[meetings_df['NoteSegment'] == seg_id].groupby('HubSpotId')['MeetingNotesPreview'].first().reset_index()
        segment_notes.columns = ['HubSpotId', f'Segment_{seg_id}_SampleNote']
        pivot_meetings_df = pivot_meetings_df.merge(segment_notes, on='HubSpotId', how='left')
        pivot_meetings_df[f'Segment_{seg_id}_SampleNote'] = pivot_meetings_df[f'Segment_{seg_id}_SampleNote'].fillna('')

    # Notes preview from each segment (all notes aggregated per contact)
    for seg_id in [-1, 0, 1, 2]:
        def aggregate_notes(group):
            notes = group['MeetingNotesPreview'].dropna().head(10)
            return ' | '.join(notes.astype(str))

        segment_previews = meetings_df[meetings_df['NoteSegment'] == seg_id].groupby('HubSpotId').apply(aggregate_notes).reset_index()
        segment_previews.columns = ['HubSpotId', f'Segment_{seg_id}_NotesPreview']
        pivot_meetings_df = pivot_meetings_df.merge(segment_previews, on='HubSpotId', how='left')
        pivot_meetings_df[f'Segment_{seg_id}_NotesPreview'] = pivot_meetings_df[f'Segment_{seg_id}_NotesPreview'].fillna('')

    # Segment descriptions (based on cluster characteristics)
    segment_descriptions = {
        -1: 'Unclustered or insufficient notes',
        0: 'General meeting notes - Cluster 0',
        1: 'General meeting notes - Cluster 1',
        2: 'General meeting notes - Cluster 2'
    }

    for seg_id in [-1, 0, 1, 2]:
        pivot_meetings_df[f'Segment_{seg_id}_Description'] = segment_descriptions.get(seg_id, 'Unknown segment')

    logging.info(f'Aggregated meeting features for {len(pivot_meetings_df)} contacts')

    # Process calls data with NLP (similar to meetings)
    logging.info('Processing calls data with NLP...')
    calls_df.rename(columns={'AssociatedContactId': 'HubSpotId'}, inplace=True)
    calls_df = calls_df[calls_df['HubSpotId'].isin(processed_df['HubSpotId'])]

    calls_df['CreateDate'] = pd.to_datetime(calls_df['CreateDate'])
    calls_df['ActivityDate'] = pd.to_datetime(calls_df['ActivityDate'])

    # Clean call notes
    calls_df['CallNotes_clean'] = calls_df['CallNotesPreview'].astype(str).apply(clean_text)
    calls_df['HasCallNotes'] = calls_df['CallNotes_clean'].str.strip().astype(bool)

    # Try Sentence-BERT embeddings first (better semantic representation)
    sbert_model = get_sbert_model()
    if sbert_model is not None:
        logging.info('Generating Sentence-BERT embeddings for call notes...')
        embeddings_calls = sbert_model.encode(
            calls_df['CallNotes_clean'].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True
        )
        logging.info(f'SBERT call embeddings generated: shape {embeddings_calls.shape}')
    else:
        # Fallback to TF-IDF if SBERT unavailable
        logging.info('Using TF-IDF embeddings for calls (SBERT not available)')
        tfidf_vectorizer_calls = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.9, ngram_range=(1, 2))
        tfidf_matrix_calls = tfidf_vectorizer_calls.fit_transform(calls_df['CallNotes_clean'])
        embeddings_calls = normalize(tfidf_matrix_calls).toarray()

    # Add call features
    calls_df['DaysFromCreateToActivity'] = (calls_df['ActivityDate'] - calls_df['CreateDate']).dt.days
    calls_df['DaysSinceCall'] = (pd.Timestamp.now() - calls_df['ActivityDate']).dt.days
    calls_df['DaysSinceCreate'] = (pd.Timestamp.now() - calls_df['CreateDate']).dt.days

    # Clustering calls
    logging.info('Clustering call notes...')
    calls_df['CallNoteSegment'] = -1

    valid_call_notes_mask = calls_df['CallNotesPreview'].notna() & (calls_df['CallNotesPreview'].str.strip() != '')
    valid_call_indices = calls_df[valid_call_notes_mask].index.tolist()

    cluster_model_calls = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=3584)
    cluster_labels_calls = cluster_model_calls.fit_predict(embeddings_calls)

    num_to_assign_calls = min(len(cluster_labels_calls), len(valid_call_indices))
    indices_to_use_calls = valid_call_indices[:num_to_assign_calls]
    labels_to_use_calls = cluster_labels_calls[:num_to_assign_calls]

    for idx, label in zip(indices_to_use_calls, labels_to_use_calls):
        calls_df.at[idx, 'CallNoteSegment'] = int(label)

    logging.info(f'Assigned {num_to_assign_calls} call cluster labels')

    # Aggregate calls by contact
    call_seg_count = calls_df.groupby(['HubSpotId', 'CallNoteSegment']).size().unstack(fill_value=0)
    call_seg_count.columns = [f"CallSegmentCount_{int(c)}" for c in call_seg_count.columns]
    call_seg_count = call_seg_count.reset_index()

    pivot_calls_df = calls_df.groupby('HubSpotId').agg(
        CallAttempts=('callDisposition', 'count'),
        ConnectedCalls=('callDisposition', lambda x: (x == 'Connected').sum()),
        NoAnswerCalls=('callDisposition', lambda x: (x == 'No Answer').sum()),
        VoicemailCalls=('callDisposition', lambda x: (x == 'Left Voicemail').sum()),
        BusyCalls=('callDisposition', lambda x: (x == 'Busy').sum()),
        AvgDaysFromCreateToActivity=('DaysFromCreateToActivity', 'mean'),
        MedianDaysFromCreateToActivity=('DaysFromCreateToActivity', 'median'),
        DaysSinceLastCall=('DaysSinceCall', 'min'),
        DaysSinceFirstCall=('DaysSinceCall', 'max'),
        FirstCallDate=('ActivityDate', 'min'),
        LastCallDate=('ActivityDate', 'max'),
        CallSpanDays=('ActivityDate', lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
    ).reset_index()

    pivot_calls_df['CallsPerMonth'] = pivot_calls_df.apply(
        lambda row: (row['CallAttempts'] / (row['CallSpanDays'] / 30.44))
        if row['CallSpanDays'] > 0 else row['CallAttempts'],
        axis=1
    )
    pivot_calls_df['ConnectionRate'] = (
        pivot_calls_df['ConnectedCalls'] / pivot_calls_df['CallAttempts']
    ).fillna(0)

    pivot_calls_df = pivot_calls_df.merge(call_seg_count, on='HubSpotId', how='left').fillna(0)

    # Add advanced call segment features
    logging.info('Creating advanced call segment features...')

    # TopCallSegmentId and TopCallSegmentLabel
    def get_top_call_segment(group):
        """Get the most common call segment for each contact"""
        if 'CallNoteSegment' in group.columns:
            segment_counts = group['CallNoteSegment'].value_counts()
            if len(segment_counts) > 0:
                top_segment = segment_counts.index[0]
                return pd.Series({
                    'TopCallSegmentId': int(top_segment),
                    'TopCallSegmentLabel': f'CallSegment_{int(top_segment)}'
                })
        return pd.Series({'TopCallSegmentId': -1, 'TopCallSegmentLabel': 'No_CallSegment'})

    top_call_segments = calls_df.groupby('HubSpotId').apply(get_top_call_segment).reset_index()
    pivot_calls_df = pivot_calls_df.merge(top_call_segments, on='HubSpotId', how='left')
    pivot_calls_df['TopCallSegmentId'] = pivot_calls_df['TopCallSegmentId'].fillna(-1).astype(int)
    pivot_calls_df['TopCallSegmentLabel'] = pivot_calls_df['TopCallSegmentLabel'].fillna('No_CallSegment')

    # Sample notes from each call segment (first note from each segment per contact)
    for seg_id in [-1, 0, 1, 2]:
        segment_notes = calls_df[calls_df['CallNoteSegment'] == seg_id].groupby('HubSpotId')['CallNotesPreview'].first().reset_index()
        segment_notes.columns = ['HubSpotId', f'CallSegment_{seg_id}_SampleNote']
        pivot_calls_df = pivot_calls_df.merge(segment_notes, on='HubSpotId', how='left')
        pivot_calls_df[f'CallSegment_{seg_id}_SampleNote'] = pivot_calls_df[f'CallSegment_{seg_id}_SampleNote'].fillna('')

    # Notes preview from each call segment (all notes aggregated per contact)
    for seg_id in [-1, 0, 1, 2]:
        def aggregate_call_notes(group):
            notes = group['CallNotesPreview'].dropna().head(10)
            return ' | '.join(notes.astype(str))

        segment_previews = calls_df[calls_df['CallNoteSegment'] == seg_id].groupby('HubSpotId').apply(aggregate_call_notes).reset_index()
        segment_previews.columns = ['HubSpotId', f'CallSegment_{seg_id}_NotesPreview']
        pivot_calls_df = pivot_calls_df.merge(segment_previews, on='HubSpotId', how='left')
        pivot_calls_df[f'CallSegment_{seg_id}_NotesPreview'] = pivot_calls_df[f'CallSegment_{seg_id}_NotesPreview'].fillna('')

    # Call segment descriptions (based on cluster characteristics)
    call_segment_descriptions = {
        -1: 'Unclustered or insufficient call notes',
        0: 'General call notes - Cluster 0',
        1: 'General call notes - Cluster 1',
        2: 'General call notes - Cluster 2'
    }

    for seg_id in [-1, 0, 1, 2]:
        pivot_calls_df[f'CallSegment_{seg_id}_Description'] = call_segment_descriptions.get(seg_id, 'Unknown segment')

    logging.info(f'Aggregated call features for {len(pivot_calls_df)} contacts')

    # Merge all features
    logging.info('Merging all features...')
    processed_df = processed_df.merge(pivot_meetings_df, on='HubSpotId', how='left')
    processed_df = processed_df.merge(pivot_calls_df, on='HubSpotId', how='left')

    # Fill NaN values from merges
    processed_df = processed_df.fillna(0)

    # Create target variable
    logging.info('Creating target variable...')
    target_mapping = {
        'SAL 2': 0,
        'SAL 1': 0,
        'Client': 1
    }

    # Map LeadStageNewCEO to Target BEFORE we need it
    if 'LeadStageNewCEO' in processed_df.columns:
        processed_df['Target'] = processed_df['LeadStageNewCEO'].map(target_mapping)
        logging.info(f"Target distribution: {processed_df['Target'].value_counts().to_dict()}")
    else:
        logging.error("LeadStageNewCEO column not found!")
        raise ValueError("LeadStageNewCEO column missing")

    # Date feature engineering
    logging.info('Engineering date features...')
    date_cols = [
        "LastContactedDate", "ProspectJourneySignUpDate", "RmNotificationDate",
        "GAA23EntryDate", "InvestorGoalAccessedDate",
        "InvestorGoalCompletedDate", "InvestorProfileAccessedDate",
        "InvestorProfileCompletedDate", "RiskAssessmentAccessedDate",
        "RiskAssessmentCompletedDate", "FollowUpDate", "FirstMeetingDate",
        "LastMeetingDate", "FirstCallDate", "LastCallDate"
    ]

    for col in date_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            processed_df[f'{col}_days'] = (pd.Timestamp.now() - processed_df[col]).dt.days
            processed_df[f'{col}_days'] = processed_df[f'{col}_days'].fillna(999)
            processed_df = processed_df.drop(columns=[col])

    logging.info(f'Final dataset shape: {processed_df.shape}')

    # Remove duplicate columns (case-insensitive) - SQL Server column names are case-insensitive
    original_cols = len(processed_df.columns)
    processed_df = processed_df.loc[:, ~processed_df.columns.str.lower().duplicated()]
    if len(processed_df.columns) < original_cols:
        logging.info(f'Removed {original_cols - len(processed_df.columns)} duplicate columns (case-insensitive)')

    # Save to SQL
    logging.info(f'Saving training dataset to {INPUT_SCHEMA}.{OUTPUT_TABLE}...')

    # Save in chunks to handle large data
    chunk_size = 10000
    for i in range(0, len(processed_df), chunk_size):
        chunk = processed_df.iloc[i:i+chunk_size]
        if i == 0:
            chunk.to_sql(name=OUTPUT_TABLE, con=target_engine, schema=INPUT_SCHEMA, if_exists='replace', index=False, method=None)
        else:
            chunk.to_sql(name=OUTPUT_TABLE, con=target_engine, schema=INPUT_SCHEMA, if_exists='append', index=False, method=None)
        logging.info(f'Saved chunk {i//chunk_size + 1}/{(len(processed_df)-1)//chunk_size + 1}')

    logging.info(f'Successfully saved {len(processed_df)} rows to {INPUT_SCHEMA}.{OUTPUT_TABLE}')
    logging.info('='*70)
    logging.info('CLIENT CONVERSION DATA PREPARATION COMPLETED SUCCESSFULLY')
    logging.info('='*70)
