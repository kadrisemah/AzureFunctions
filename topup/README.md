# Top-Up Prediction Pipeline

Azure Functions for predicting client top-up probability using CatBoost ML model.

## Functions

1. **data_preparation** - Extracts HubSpot data, creates monthly panel dataset
2. **model_training** - Trains CatBoost model, generates predictions

## Schedule

- **Data Preparation:** 1st of month at 6:00 AM UTC
- **Model Training:** 1st of month at 6:30 AM UTC

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
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/data_preparation -Method POST -ContentType "application/json" -Body "{}"
Invoke-RestMethod -Uri http://localhost:7071/admin/functions/model_training -Method POST -ContentType "application/json" -Body "{}"
```

## Outputs

- `dbo.topup_training_dataset` - Training data (~50,000 rows)
- `dbo.topup_model_predictions` - Predictions for all clients
- `dbo.topup_rm_action_list` - High-probability clients
- `dbo.topup_feature_importance` - Feature rankings

## Expected Duration

- Data Preparation: ~10 minutes
- Model Training: ~20 minutes
