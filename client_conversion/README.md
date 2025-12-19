# Client Conversion Prediction Pipeline

## Overview

This Azure Functions pipeline predicts which SAL 1/2 prospects will convert to Clients (Committed Capital stage). It uses machine learning (CatBoost + Optuna) with NLP analysis of meeting and call notes to identify high-probability prospects for relationship managers.

## Business Value

**Pipeline Position**: SAL → Client (Late Funnel)

- **Goal**: Predict which SAL prospects will close and become revenue-generating clients
- **Use Case**: Help RMs prioritize SAL prospects most likely to convert
- **Output**: Ranked list of high-probability SAL prospects with conversion scores

## Pipeline Architecture

### Function 1: `client_conversion_data_preparation`
**Schedule**: 1st of month, 8:00 AM
**Runtime**: ~20-30 minutes

**What it does:**
1. Extracts Client + SAL data from vi_HSContact
2. Extracts meeting history (vi_HSEngagementmeeting)
3. Extracts call history (vi_HSEngagementCall)
4. NLP processing:
   - TF-IDF vectorization of meeting/call notes
   - MiniBatchKMeans clustering (n_clusters=3)
   - Semantic segmentation of engagement
5. Feature engineering:
   - 290+ contact features
   - Meeting metrics (volume, frequency, completion rate)
   - Call metrics (attempts, connection rate)
   - Date transformations (days since events)
6. Saves training dataset to `dbo.client_conversion_training_dataset`

### Function 2: `client_conversion_model_training`
**Schedule**: 1st of month, 8:30 AM
**Runtime**: ~40-60 minutes

**What it does:**
1. Loads training dataset
2. Optuna hyperparameter optimization (50 trials)
3. 5-fold Stratified Cross-Validation
4. Out-of-fold (OOF) predictions for all data
5. Trains final CatBoost model on all data
6. Generates predictions with probabilities
7. Creates RM action list (probability ≥ 0.5)
8. Saves results to multiple tables

## Output Tables

| Table | Description | Rows |
|-------|-------------|------|
| `client_conversion_training_dataset` | Processed features + target | ~2,800 |
| `client_conversion_model_predictions` | All predictions with OOF scores | ~2,800 |
| `client_conversion_rm_action_list` | High-probability SAL prospects | ~500-1,000 |
| `client_conversion_feature_importance` | Top predictive features | ~300 |
| `client_conversion_model_metadata` | Model performance metrics | 1 per run |
| `client_conversion_optuna_study` | Hyperparameter tuning history | 50 per run |

## Key Features

### Target Variable
```python
'SAL 1': 0    # Has NOT converted
'SAL 2': 0    # Has NOT converted
'Client': 1   # HAS converted (Committed Capital)
```

### Model Configuration
- **Algorithm**: CatBoostClassifier
- **Optimization**: Optuna with TPESampler
- **Validation**: 5-fold Stratified CV
- **Metric**: ROC-AUC
- **Performance**: ~95% accuracy, ~88% F1-score

### NLP Processing
- **TF-IDF**: max_features=5000, ngram_range=(1,2), min_df=5, max_df=0.9
- **Clustering**: MiniBatchKMeans, n_clusters=3, batch_size=3584
- **Text Cleaning**: Stopwords removal, lemmatization, contraction expansion

## Local Development

### Prerequisites
- Python 3.11
- Azure Functions Core Tools 4.x
- Azurite
- ODBC Driver 18 for SQL Server

### Setup
```bash
cd client_conversion

# Create virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp local.settings.json.template local.settings.json
# Edit local.settings.json with your database credentials

# Start Azurite (separate terminal)
azurite

# Start functions
func start
```

### Manual Trigger (PowerShell)
```powershell
# Data Preparation
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/client_conversion_data_preparation -Method POST -ContentType "application/json" -Body "{}"

# Model Training (wait for data prep to complete)
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/client_conversion_model_training -Method POST -ContentType "application/json" -Body "{}"
```

## Azure Deployment

```bash
# Login
az login

# Create Function App (if needed)
az functionapp create \
  --resource-group <YOUR_RESOURCE_GROUP> \
  --name <YOUR_FUNCTION_APP_NAME> \
  --storage-account <YOUR_STORAGE_ACCOUNT> \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux

# Deploy
func azure functionapp publish <YOUR_FUNCTION_APP_NAME> --python

# Configure credentials
az functionapp config appsettings set \
  --name <YOUR_FUNCTION_APP_NAME> \
  --resource-group <YOUR_RESOURCE_GROUP> \
  --settings \
    SOURCE_SERVER="..." \
    SOURCE_DATABASE="..." \
    SOURCE_USER="..." \
    SOURCE_PASSWORD="..." \
    TARGET_SERVER="..." \
    TARGET_DATABASE="..." \
    TARGET_USER="..." \
    TARGET_PASSWORD="..."
```

## Monitoring

### Key Metrics to Track
- Execution time (target: <2 hours total)
- OOF ROC-AUC score (target: >0.85)
- High-probability contact count
- Feature importance stability

### Logs Location
- Azure Portal → Function App → Functions → Monitor
- Application Insights (if configured)

## Relationship to Other Pipelines

```
Customer Journey:
Lead → SQL 1/2 → SAL 1/2 → CLIENT
         ↑            ↑
   SQLSAL Pipeline  THIS Pipeline
```

- **Top-Up Pipeline**: Predicts existing client top-up probability
- **SQLSAL Pipeline**: Predicts SQL → SAL conversion
- **This Pipeline**: Predicts SAL → Client conversion

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'nltk'"
**Solution**: NLTK data is downloaded automatically on first run

### Issue: "Cannot connect to database"
**Solution**: Verify firewall rules allow your IP, check credentials

### Issue: "Memory issues during training"
**Solution**: Reduce N_OPTUNA_TRIALS or use smaller batch_size

### Issue: "TF-IDF fails on empty notes"
**Solution**: Pipeline handles empty notes automatically with clean_text()

## Technical Details

### Data Sources
- **vi_HSContact**: Client demographics + SAL prospects
- **vi_HSEngagementmeeting**: Meeting history + notes
- **vi_HSEngagementCall**: Call history + notes

### Dependencies
- catboost==1.2.2 (ML model)
- optuna==3.5.0 (hyperparameter optimization)
- nltk==3.8.1 (NLP preprocessing)
- scikit-learn==1.3.0 (TF-IDF, clustering)
- pandas==2.0.3, numpy==1.24.3 (data processing)

### Performance Tuning
- Chunk size for SQL writes: 10,000 rows
- Optuna trials: 50 (configurable)
- Cross-validation folds: 5
- Early stopping rounds: 50


-------

