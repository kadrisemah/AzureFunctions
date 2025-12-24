# Top-Up Prediction Pipeline - Technical Documentation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Data Sources](#data-sources)
4. [Pipeline Components](#pipeline-components)
5. [Implementation Details](#implementation-details)
6. [Configuration](#configuration)
7. [Output Tables](#output-tables)
8. [Dependencies](#dependencies)
9. [Deployment](#deployment)
10. [Testing](#testing)
11. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Executive Summary

### Purpose
The Top-Up Prediction Pipeline is an Azure Functions-based ML system that predicts which existing clients are likely to increase their investment commitments (top-ups) and provides AI-generated actionable insights for Relationship Managers.

### Key Features
- **Predictive Modeling**: CatBoost classifier trained on 67+ enriched client features
- **Explainability**: SHAP value analysis for model transparency
- **AI-Powered Insights**: GPT-4 generated personalized pitches for high-probability clients
- **Client Segmentation**: KMeans clustering (4 segments) for behavioral profiling
- **Automated Execution**: Monthly scheduled runs with complete automation

### Business Value
- **Proactive Client Engagement**: Identifies top-up opportunities before they happen
- **RM Efficiency**: Provides ready-to-use talking points and client insights
- **Data-Driven Decisions**: Removes guesswork with probability scores and explanations
- **Scalability**: Processes 800+ clients automatically each month

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MONTHLY EXECUTION                           │
│                      (1st of Month, 6:00 AM)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FUNCTION 1: data_preparation                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. Connect to PrimeAI Database                               │  │
│  │  2. Fetch 67 enriched features from PrimeAiMandates           │  │
│  │  3. Perform KMeans clustering (4 segments)                    │  │
│  │  4. Create SegmentClass one-hot encoded features              │  │
│  │  5. Save to dbo.topup_training_dataset                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (30 min wait)
┌─────────────────────────────────────────────────────────────────────┐
│  FUNCTION 2: model_training                                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. Load training data from dbo.topup_training_dataset        │  │
│  │  2. Feature engineering & outlier detection                   │  │
│  │  3. Train CatBoost classifier (5000 iterations)               │  │
│  │  4. Generate predictions for all clients                      │  │
│  │  5. For high-probability clients (≥50%):                      │  │
│  │     a. Generate SHAP explanations                             │  │
│  │     b. Generate AI pitches (GPT-4)                            │  │
│  │  6. Save results to 3 tables                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT TABLES                               │
│  • dbo.topup_model_predictions (all clients)                        │
│  • dbo.topup_rm_action_list (high-prob + explanations)              │
│  • dbo.topup_feature_importance (feature rankings)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Runtime** | Azure Functions | 4.x |
| **Language** | Python | 3.11 |
| **ML Framework** | CatBoost | 1.2.2 |
| **Explainability** | SHAP (CatBoost native) | Built-in |
| **AI Generation** | Azure OpenAI GPT-4 | gpt-4o |
| **Clustering** | scikit-learn KMeans | 1.3.0 |
| **Database** | Azure SQL Server | - |
| **Orchestration** | Timer Triggers (CRON) | - |

---

## Data Sources

### Primary Database: PrimeAI (Feature Source)

**Connection Details:**
- **Server:** `tfo-az-prod-dbsr01.database.windows.net`
- **Database:** `sqldb-tfo-primeai-prod`
- **Table:** `dbo.PrimeAiMandates`
- **Purpose:** Enriched client features with AI-generated profiles

**Table Structure:**
- **Rows:** ~800-1,000 (one per client)
- **Columns:** 67 features
- **Update Frequency:** Monthly (pre-processed by upstream pipeline)

**Key Features Retrieved:**

1. **Business Profile (14 features)**
   - ContactType, LeadStageNew, Nationality, MaritalStatus
   - EmploymentField, ShariaCompliant, AgeRange, TypeOfReferral
   - Gender, wealth_range, engagement_label, client_approached_us
   - share_of_wallet_label, digital_campaign_influence

2. **Engagement Metrics (10 features)**
   - TotalWebinarsAttended, TotalWebinarsRegistered
   - MarketingEmailsClicked, MarketingEmailsOpened, MarketingEmailsDelivered
   - TotalCallsConnected, TotalCallsMade
   - meetscounts, callscountsconnected
   - opened_counts, clicked_counts, delivered_counts

3. **Aggregated Text/Notes (4 features)**
   - MEETINGNOTESAGGREGATED
   - CALLNOTESAGGREGATED
   - meetingnotespreview, callnotespreview
   - emailbodypreview, maildata

4. **Digital Activity (8 features)**
   - FORMS_FILLED, numberofformsfilled
   - WEBSITEVISTED, numberofwebvisits
   - CLIENTAPP, numberofclientappvisits
   - PJVISITED, PJVISITS

5. **Top-Up History (4 features)**
   - Total Top Ups, Total Top Up Amount
   - Top Up Frequency, they_topped_up

6. **Financial/Commitment (2 features)**
   - total_commitment, First Commitment Amount
   - DEALCLOSEDATECOMMITTEDCAPITAL

7. **Asset Class Features (8 features)**
   - AssetClass_Cash & Cash Equivalent
   - AssetClass_Credit
   - AssetClass_Equities
   - AssetClass_Multi-Asset Class
   - AssetClass_Others
   - AssetClass_Private Equity
   - AssetClass_Real Estate
   - AssetClass_Yielding Investments

8. **Sub-Strategy Features (5 features)**
   - SubStrategy_Absolute Return
   - SubStrategy_Capital Growth
   - SubStrategy_Capital Yielding
   - SubStrategy_Opportunistic
   - SubStrategy_Others

9. **Metadata (5 features)**
   - Activities per Quarter
   - Hubspot_Persona
   - SegmentClass (overwritten by KMeans)
   - RM_Name
   - DealCloseDateCommittedCapitalContact

10. **AI-Generated Profiles (3 features)**
    - Client Summary (GPT-4 generated client overview)
    - Client Empathy Map (Understanding client perspective)
    - Psychology Profile (Behavioral insights)

### Target Database: DWH (Results Storage)

**Connection Details:**
- **Server:** `tfo-az-prod-dwh-sqlsrv-003.database.windows.net`
- **Database:** `tfo-az-prod-dwh-sqldb-003`
- **Schema:** `dbo`
- **Purpose:** Store training data, predictions, and action lists

---

## Pipeline Components

### Component 1: Data Preparation Function

**File:** `topup/data_preparation/__init__.py`
**Trigger:** Timer - `0 0 6 1 * *` (6:00 AM, 1st of month)
**Duration:** ~10 minutes

#### Process Flow

1. **Database Connection**
   ```python
   # PrimeAI connection
   primeai_engine = create_engine(
       f'mssql+pyodbc://{user}:{password}@{server}:1433/{database}'
   )
   ```

2. **Feature Extraction**
   - Executes SQL query to fetch all 67 features
   - Loads into pandas DataFrame (~800 rows)

3. **Client Segmentation (KMeans)**
   ```python
   def perform_monthly_clustering(df, n_clusters=4):
       # Select numeric features (exclude IDs, text)
       numeric_features = df.select_dtypes(include=[np.number]).columns

       # Perform clustering
       kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
       df['SegmentClass'] = kmeans.fit_predict(X_cluster)

       # Create one-hot encoded features
       for i in range(4):
           df[f'SegmentClass_{i}'] = (df['SegmentClass'] == i).astype(int)

       return df
   ```

4. **Metadata Addition**
   - `created_at`: Timestamp
   - `pipeline_run_id`: Run identifier (YYYYMMDD_HHMMSS)

5. **Save to Database**
   - **Table:** `dbo.topup_training_dataset`
   - **Method:** Drop and recreate (if_exists='replace')
   - **Chunking:** 1000 rows per batch
   - **Schema:** Auto-inferred from DataFrame

#### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_clusters` | 4 | Number of client segments |
| `random_state` | 42 | Reproducibility |
| `n_init` | 10 | KMeans initialization attempts |
| `chunk_size` | 1000 | SQL insert batch size |

#### Output

**Table:** `dbo.topup_training_dataset`

| Column Type | Count | Examples |
|-------------|-------|----------|
| **Original Features** | 67 | ContactType, Total Top Ups, Client Summary |
| **Clustering Features** | 5 | SegmentClass, SegmentClass_0-3 |
| **Metadata** | 2 | created_at, pipeline_run_id |
| **Total Columns** | 74 | - |
| **Row Count** | ~800-1,000 | One per client |

---

### Component 2: Model Training Function

**File:** `topup/model_training/__init__.py`
**Trigger:** Timer - `0 30 6 1 * *` (6:30 AM, 1st of month)
**Duration:** ~20-30 minutes

#### Process Flow

##### Stage 1: Data Loading & Preparation

1. **Load Training Data**
   ```python
   data = pd.read_sql(
       f'SELECT * FROM dbo.topup_training_dataset',
       engine
   )
   ```

2. **Feature Selection**
   - Exclude columns: HubSpotId, TOPUP, TOPUPAMOUNT, metadata, text columns
   - Numeric features: ~60-65 features
   - Categorical features: Encode with LabelEncoder

3. **Target Variable Creation**
   ```python
   data['target_flag'] = (data['they_topped_up'].fillna(0) > 0).astype(int)
   ```

##### Stage 2: Outlier Detection

```python
iso_forest = IsolationForest(
    contamination=0.05,  # Expect 5% outliers
    random_state=42,
    n_jobs=-1
)
outlier_labels = iso_forest.fit_predict(X_outlier)
train_df = train_df[train_df['is_outlier'] == 0]  # Remove outliers
```

##### Stage 3: Model Training (CatBoost)

**Hyperparameters:**
```python
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
```

**Why CatBoost?**
- Native categorical feature handling
- Built-in SHAP value calculation
- Robust to overfitting
- Excellent performance on tabular data

##### Stage 4: Prediction Generation

1. **Generate Predictions for ALL Clients**
   ```python
   y_proba_all = model.predict_proba(X_all)[:, 1]
   ```

2. **Merge with PrimeAI Metadata**
   - Fetches Client Summary, Empathy Map, Psychology Profile
   - Adds Top Up Frequency, Activities per Quarter, RM_Name

##### Stage 5: Explainability & AI Pitch Generation

**For clients with probability ≥ 50%:**

1. **SHAP Explanation**
   ```python
   def explain_catboost_client_savefig(hid, X_test, model, categorical_feats):
       # Get SHAP values using CatBoost native method
       shap_vec = model.get_feature_importance(pool, type="ShapValues")[0]

       # Extract feature contributions
       shap_feat = shap_vec[:-1]
       base_val = shap_vec[-1]

       # Create DataFrame: Feature | Value | SHAP
       df = pd.DataFrame({
           'Feature': feature_names,
           'Value': display_vals,
           'SHAP': shap_values
       }).sort_values(by='SHAP', key=abs, ascending=False)

       return df
   ```

2. **AI Pitch Generation (GPT-4)**
   ```python
   def generate_pitch_with_shap(client_summary, probability, shap_explanation):
       pitch_prompt = """
       You are a Senior Financial Advisor at a prestigious family office.

       Your role is to clearly and persuasively explain AI-driven top-up
       predictions to a Relationship Manager (RM) who works directly with clients.

       Please:
       - Analyze the attached SHAP summary
       - Explain in plain business language how each detected variable contributes
       - Link explanation to the model's TOPUP probability
       - Focus on what RM should understand or tell the client
       - Close with summary explaining why probability is what it is

       Do NOT use technical jargon. Assume RM has no technical background.
       """

       # Call Azure OpenAI GPT-4
       response = requests.post(AZURE_OPENAI_ENDPOINT, ...)
       return response['choices'][0]['message']['content']
   ```

##### Stage 6: Output Generation

**Three tables created:**

1. **`dbo.topup_model_predictions`** - All clients
2. **`dbo.topup_rm_action_list`** - High-probability clients only
3. **`dbo.topup_feature_importance`** - Model insights

---

## Implementation Details

### Critical Code Sections

#### 1. Dynamic Table Creation

```python
def infer_sqlalchemy_type(dtype, column_name=None, series=None):
    """Infer SQLAlchemy type from pandas dtype"""
    if 'datetime' in str(dtype):
        return DateTime
    if str(dtype).startswith('int'):
        return Integer
    if str(dtype).startswith('float'):
        return Float
    if str(dtype) == 'bool':
        return Boolean
    if str(dtype) == 'object':
        # Special handling for long text columns
        long_text_cols = ['meetingnotesaggregated', 'client_summary', ...]
        if column_name and any(kw in column_name.lower() for kw in long_text_cols):
            return NVARCHAR(None)  # NVARCHAR(MAX)
        # Dynamic sizing based on max length
        max_len = series.astype(str).str.len().max()
        if max_len > 4000:
            return NVARCHAR(None)
        elif max_len > 500:
            return NVARCHAR(4000)
        else:
            return NVARCHAR(min(max(int(max_len * 1.5), 50), 500))
    return NVARCHAR(500)
```

#### 2. KMeans Clustering Logic

```python
def perform_monthly_clustering(df, n_clusters=4):
    # Select only numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID columns and existing SegmentClass
    exclude_cols = ['Client_ID', 'HubSpotId', 'OnboardingId', 'SegmentClass']
    numeric_features = [col for col in numeric_features if col not in exclude_cols]

    if len(numeric_features) > 0:
        X_cluster = df[numeric_features].fillna(0)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['SegmentClass'] = kmeans.fit_predict(X_cluster)

        # One-hot encode
        for i in range(n_clusters):
            df[f'SegmentClass_{i}'] = (df['SegmentClass'] == i).astype(int)

        logging.info(f'Segment distribution: {df["SegmentClass"].value_counts().to_dict()}')

    return df
```

#### 3. SHAP Value Extraction

```python
# CatBoost native SHAP implementation
row = X_test.loc[[hid]]
cat_features_idx = [row.columns.get_loc(c) for c in categorical_feats if c in row.columns]
pool = Pool(row, cat_features=cat_features_idx)

# Get SHAP vector [feature_shap_1, feature_shap_2, ..., base_value]
shap_vec = model.get_feature_importance(pool, type="ShapValues")[0]
shap_feat = shap_vec[:-1]  # Feature contributions
base_val = shap_vec[-1]    # Model base value
```

#### 4. Azure OpenAI Integration

```python
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": pitch_prompt},
                {"type": "text", "text": f"Client Summary: {client_summary}"},
                {"type": "text", "text": f"AI Model Probability: {probability:.2f}"},
                {"type": "text", "text": f"SHAP Explanation: {shap_df}"}
            ]
        }
    ],
    "max_tokens": 500
}

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_KEY
}

response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
```

---

## Configuration

### Environment Variables

**File:** `local.settings.json` (local) / Application Settings (Azure)

```json
{
  "TARGET_SERVER": "tfo-az-prod-dwh-sqlsrv-003.database.windows.net",
  "TARGET_DATABASE": "tfo-az-prod-dwh-sqldb-003",
  "TARGET_USER": "data.science",
  "TARGET_PASSWORD": "***",

  "PRIMEAI_SERVER": "tfo-az-prod-dbsr01.database.windows.net",
  "PRIMEAI_DATABASE": "sqldb-tfo-primeai-prod",
  "PRIMEAI_USER": "aiprime",
  "PRIMEAI_PASSWORD": "***",

  "AZURE_OPENAI_ENDPOINT": "https://eastusipms.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
  "AZURE_OPENAI_KEY": "***"
}
```

### Scheduling (CRON Expressions)

| Function | Schedule | CRON | Description |
|----------|----------|------|-------------|
| **data_preparation** | Monthly, 6:00 AM | `0 0 6 1 * *` | 1st of month, 6AM UTC |
| **model_training** | Monthly, 6:30 AM | `0 30 6 1 * *` | 1st of month, 6:30AM UTC |

**CRON Format:** `{second} {minute} {hour} {day} {month} {day-of-week}`

---

## Output Tables

### Table 1: `dbo.topup_training_dataset`

**Purpose:** Prepared training data with clustering features
**Created By:** data_preparation function
**Row Count:** ~800-1,000 (one per client)

**Schema:**
```sql
CREATE TABLE dbo.topup_training_dataset (
    -- Original PrimeAI features (67 columns)
    [Client ID] NVARCHAR(50),
    ContactType NVARCHAR(255),
    LeadStageNew NVARCHAR(255),
    -- ... (64 more columns)
    [Client Summary] NVARCHAR(MAX),
    [Client Empathy Map] NVARCHAR(MAX),
    [Psychology Profile] NVARCHAR(MAX),

    -- Clustering features (5 columns)
    SegmentClass INT,
    SegmentClass_0 INT,
    SegmentClass_1 INT,
    SegmentClass_2 INT,
    SegmentClass_3 INT,

    -- Metadata (2 columns)
    created_at DATETIME,
    pipeline_run_id NVARCHAR(50)
);
```

### Table 2: `dbo.topup_model_predictions`

**Purpose:** All client predictions with metadata
**Created By:** model_training function
**Row Count:** ~800-1,000 (all clients)

**Schema:**
```sql
CREATE TABLE dbo.topup_model_predictions (
    HubSpotId BIGINT,
    TOPUP_PROBABILITY_THIS_MONTH FLOAT,
    [Top Up Frequency] FLOAT,
    they_topped_up INT,
    [Activities per Quarter] FLOAT,
    Hubspot_Persona NVARCHAR(255),
    RM_Name NVARCHAR(255),
    [Client Summary] NVARCHAR(MAX),
    [Client Empathy Map] NVARCHAR(MAX),
    [Psychology Profile] NVARCHAR(MAX)
);
```

**Sample Row:**
| HubSpotId | TOPUP_PROBABILITY | Top Up Frequency | Client Summary |
|-----------|-------------------|------------------|----------------|
| 82155 | 0.758779 | 2.5 | "Abdullah is a high-net-worth..." |

### Table 3: `dbo.topup_rm_action_list`

**Purpose:** High-probability clients with AI explanations
**Created By:** model_training function
**Row Count:** ~200-300 (clients with probability ≥ 0.5)

**Schema:**
```sql
CREATE TABLE dbo.topup_rm_action_list (
    HubSpotId BIGINT,
    TOPUP_PROBABILITY_THIS_MONTH FLOAT,
    [Top Up Frequency] FLOAT,
    they_topped_up INT,
    [Activities per Quarter] FLOAT,
    Hubspot_Persona NVARCHAR(255),
    RM_Name NVARCHAR(255),
    [Client Summary] NVARCHAR(MAX),
    [Client Empathy Map] NVARCHAR(MAX),
    [Psychology Profile] NVARCHAR(MAX),
    explain NVARCHAR(MAX),              -- SHAP explanation (DataFrame as text)
    pitch NVARCHAR(MAX)                 -- GPT-4 generated pitch
);
```

**Sample Row:**
| HubSpotId | TOPUP_PROBABILITY | explain | pitch |
|-----------|-------------------|---------|-------|
| 82155 | 0.758779 | "Feature Value SHAP\ntotal_commitment 250000 0.15..." | "The client shows strong potential for a top-up based on..." |

### Table 4: `dbo.topup_feature_importance`

**Purpose:** Model feature rankings
**Created By:** model_training function
**Row Count:** ~65-70 (one per feature)

**Schema:**
```sql
CREATE TABLE dbo.topup_feature_importance (
    feature NVARCHAR(255),
    importance FLOAT
);
```

**Sample Rows:**
| feature | importance |
|---------|------------|
| total_commitment | 15.234 |
| Total Top Ups | 12.891 |
| Activities per Quarter | 9.456 |

---

## Dependencies

### Python Packages (`requirements.txt`)

```
azure-functions
pandas==2.0.3
numpy==1.24.3
sqlalchemy==2.0.19
pyodbc==4.0.39
catboost==1.2.2
scikit-learn==1.3.0
matplotlib==3.7.2
python-dateutil==2.8.2
requests==2.31.0
tqdm==4.66.1
```

### System Requirements

- **Python:** 3.11 (or 3.10)
- **Azure Functions Core Tools:** 4.x
- **ODBC Driver:** 17 or 18 for SQL Server
- **Azurite:** Latest (for local development)

---

## Deployment

### Local Development

```bash
# 1. Navigate to topup folder
cd topup

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure credentials
cp local.settings.json.template local.settings.json
# Edit local.settings.json with actual credentials

# 6. Start Azurite (Terminal 1)
azurite

# 7. Start Functions (Terminal 2)
func start

# 8. Manual trigger (Terminal 3 - PowerShell)
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/data_preparation -Method POST -ContentType "application/json" -Body "{}"
```

### Azure Deployment

```bash
# 1. Login to Azure
az login

# 2. Deploy to Function App
cd topup
func azure functionapp publish YOUR_FUNCTION_APP_NAME --python

# 3. Configure Application Settings
az functionapp config appsettings set \
  --name YOUR_FUNCTION_APP_NAME \
  --resource-group YOUR_RESOURCE_GROUP \
  --settings \
    TARGET_SERVER="tfo-az-prod-dwh-sqlsrv-003.database.windows.net" \
    TARGET_DATABASE="tfo-az-prod-dwh-sqldb-003" \
    TARGET_USER="data.science" \
    TARGET_PASSWORD="***" \
    PRIMEAI_SERVER="tfo-az-prod-dbsr01.database.windows.net" \
    PRIMEAI_DATABASE="sqldb-tfo-primeai-prod" \
    PRIMEAI_USER="aiprime" \
    PRIMEAI_PASSWORD="***" \
    AZURE_OPENAI_ENDPOINT="https://eastusipms.cognitiveservices.azure.com/..." \
    AZURE_OPENAI_KEY="***"
```

---

## Testing

### Pre-Deployment Checklist

- [ ] PrimeAiMandates table structure verified (run `VERIFY_PRIMEAI_STRUCTURE.sql`)
- [ ] All environment variables configured
- [ ] Database firewall allows Azure Function IP
- [ ] Azure OpenAI endpoint accessible
- [ ] Dependencies installed (`requirements.txt`)
- [ ] Local testing successful

### Test Scenarios

#### Test 1: Data Preparation

**Expected:**
- Connects to PrimeAI successfully
- Fetches ~800-1,000 rows
- Creates 4 client segments
- Saves to `dbo.topup_training_dataset`
- Execution time: <15 minutes

**Validation:**
```sql
-- Check row count
SELECT COUNT(*) FROM dbo.topup_training_dataset;

-- Check segment distribution
SELECT SegmentClass, COUNT(*) FROM dbo.topup_training_dataset GROUP BY SegmentClass;

-- Check for nulls
SELECT COUNT(*) FROM dbo.topup_training_dataset WHERE [Client Summary] IS NULL;
```

#### Test 2: Model Training

**Expected:**
- Loads training data successfully
- Trains model (5000 iterations, ~10-15 min)
- Generates predictions for all clients
- Creates ~200-300 high-probability action items
- Generates SHAP explanations
- Generates GPT-4 pitches
- Execution time: <30 minutes

**Validation:**
```sql
-- Check predictions
SELECT COUNT(*),
       AVG(TOPUP_PROBABILITY_THIS_MONTH),
       MIN(TOPUP_PROBABILITY_THIS_MONTH),
       MAX(TOPUP_PROBABILITY_THIS_MONTH)
FROM dbo.topup_model_predictions;

-- Check action list
SELECT COUNT(*) FROM dbo.topup_rm_action_list;

-- Check for explanations
SELECT COUNT(*) FROM dbo.topup_rm_action_list WHERE explain IS NOT NULL AND pitch IS NOT NULL;

-- View sample pitch
SELECT TOP 1 HubSpotId, TOPUP_PROBABILITY_THIS_MONTH, pitch FROM dbo.topup_rm_action_list ORDER BY TOPUP_PROBABILITY_THIS_MONTH DESC;
```

---

## Monitoring & Troubleshooting

### Application Insights Queries

```kusto
// Function execution history
traces
| where timestamp > ago(7d)
| where message contains "TOPUP"
| project timestamp, severityLevel, message
| order by timestamp desc

// Errors only
exceptions
| where timestamp > ago(7d)
| project timestamp, type, outerMessage, problemId

// Performance metrics
requests
| where timestamp > ago(7d)
| where name in ("data_preparation", "model_training")
| summarize avg(duration), max(duration), count() by name
```

### Common Issues

#### Issue 1: "Missing required environment variables"

**Cause:** Environment variables not configured
**Solution:**
```bash
# Check configuration
az functionapp config appsettings list --name YOUR_APP --resource-group YOUR_RG

# Set missing variables
az functionapp config appsettings set --name YOUR_APP --resource-group YOUR_RG --settings "PRIMEAI_SERVER=..."
```

#### Issue 2: "Cannot connect to database"

**Cause:** Firewall blocking Function App IP
**Solution:**
1. Get Function App outbound IPs:
   ```bash
   az functionapp show --name YOUR_APP --resource-group YOUR_RG --query "outboundIpAddresses"
   ```
2. Add IPs to SQL Server firewall rules

#### Issue 3: "No data in PrimeAiMandates"

**Cause:** Upstream pipeline not run
**Solution:**
- Verify PrimeAiMandates table has recent data
- Check upstream pipeline execution logs

#### Issue 4: "OpenAI API timeout"

**Cause:** Network latency or rate limiting
**Solution:**
- Check Azure OpenAI service health
- Verify API key is valid
- Review rate limits (TPM/RPM)

#### Issue 5: "SHAP generation failed"

**Cause:** Client not in test set or invalid features
**Solution:**
- Check logs for specific HubSpotId
- Verify feature columns match training data
- Check for null values in features

---

## Performance Metrics

### Expected Execution Times

| Function | Typical | Maximum | Notes |
|----------|---------|---------|-------|
| **data_preparation** | 8-10 min | 15 min | Depends on PrimeAI query speed |
| **model_training** | 20-25 min | 35 min | SHAP + GPT-4 generation |
| **Total Pipeline** | 30 min | 50 min | End-to-end |

### Resource Usage

- **Memory:** ~1-2 GB peak
- **CPU:** 2 cores recommended
- **Storage:** <500 MB (model file ~50 MB)
- **Network:** ~100 MB data transfer

---
