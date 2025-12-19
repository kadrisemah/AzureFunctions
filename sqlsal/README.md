# SQL/SAL Lead Scoring Pipeline

Azure Functions for scoring SQL/SAL prospects using CatBoost with Optuna optimization and NLP-based engagement analysis.

**Source:** Migrated from `SQLSAL 1.ipynb` (included in this folder)

## Functions

1. **sqlsal_data_preparation** - Extracts SQL/SAL prospects, processes engagement with NLP clustering
2. **sqlsal_model_training** - Trains CatBoost with Optuna, generates lead scores

## Schedule

- **Data Preparation:** 1st of month at 8:00 AM UTC
- **Model Training:** 1st of month at 8:30 AM UTC

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp local.settings.json.template local.settings.json
# Edit local.settings.json with your database credentials

# Start Azurite (Terminal 1)
azurite

# Start functions (Terminal 2)
func start

# Test functions (Terminal 3 - PowerShell)
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/sqlsal_data_preparation -Method POST -ContentType "application/json" -Body "{}"
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/sqlsal_model_training -Method POST -ContentType "application/json" -Body "{}"
```

## Outputs

- `dbo.sqlsal_training_dataset` - Engineered features with NLP clusters (~10,000-20,000 rows)
- `dbo.sqlsal_model_predictions` - All predictions with probabilities
- `dbo.sqlsal_rm_action_list` - High-probability SQL contacts (filtered)
- `dbo.sqlsal_feature_importance` - Ranked feature importance
- `dbo.sqlsal_model_metadata` - Training metrics & hyperparameters

## Key Features

- **NLP clustering** of meeting/call notes using TF-IDF
- **Optuna hyperparameter optimization** (50 trials)
- **One-hot encoding** of 60+ categorical features
- **100% notebook fidelity** - all parameters and logic preserved

## Expected Duration

- Data Preparation: ~15-20 minutes
- Model Training: ~30-40 minutes (Optuna optimization)

## Data Sources

- `vi_HSContact` - SQL/SAL prospects (SQL 1/2, SAL 1/2)
- `vi_HSEngagementmeeting` - Meeting engagement history
- `vi_HSEngagementCall` - Call engagement history
